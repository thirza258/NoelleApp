"""Comparative bench executor: View 1 -> View N strictly in order on the GPU lane.

Before running, the combined footprint of every selected model is checked
against the usable memory budget (budget minus the 15 % headroom):

* **retain** - everything fits: models stay resident and later runs reuse them.
* **hot_swap** - they do not: before each view, least-recently-used models are
  unloaded (``del``, ``gc.collect()``, ``empty_cache()``) until the next one fits.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable

from noelle.runtime import devices
from noelle.runtime.errors import BadRequest, NoelleRuntimeError
from noelle.runtime.model_manager import ModelManager, ModelSpec
from noelle.runtime.preflight import plan_queue
from noelle.runtime.protocol import ErrorCode, Event, GenerationParams

Emit = Callable[[Event, dict[str, Any]], None]
MAX_VIEWS = 4


def run_benchmark(manager: ModelManager, payload: dict[str, Any], *, stop_event: threading.Event,
                  emit: Emit) -> dict[str, Any]:
    views = payload.get("views") or []
    if not views:
        raise BadRequest("Select at least one model.")
    if len(views) > MAX_VIEWS:
        raise BadRequest(f"At most {MAX_VIEWS} views are supported.")
    prompt = str(payload.get("prompt") or "").strip()
    if not prompt and not all(str(v.get("prompt") or "").strip() for v in views):
        raise BadRequest("The prompt is empty.")
    system_prompt = payload.get("system_prompt") or None
    params = GenerationParams.from_dict(payload.get("params"))

    specs: dict[int, ModelSpec] = {}
    prompts: dict[int, str] = {}
    jobs: list[tuple[int, str, int]] = []
    resident = manager.resident_sizes()
    for entry in views:
        view = int(entry.get("view", len(specs) + 1))
        spec = ModelSpec.from_payload({**entry, "task": entry.get("task") or "text-generation"})
        specs[view] = spec
        prompts[view] = str(entry.get("prompt") or "").strip() or prompt  # per-view override when not broadcasting
        device = manager.resolve_device(spec)
        key = spec.key(device)
        size = resident.get(key)
        if size is None:
            source = manager.resolve_source(spec)
            size = manager.estimate_bytes(spec, source, device)
        jobs.append((view, key, size))

    budget = manager.budget()
    plan = plan_queue(jobs, budget.usable_bytes, resident=resident)
    plan_data = plan.to_dict()
    plan_data["device"] = manager.device
    emit(Event.QUEUE_PLAN, plan_data)

    results: list[dict[str, Any]] = []
    started = time.perf_counter()
    for step in plan.steps:
        if stop_event.is_set():
            results.append({"view": step.view, "status": "cancelled"})
            emit(Event.VIEW_COMPLETE, {"view": step.view, "status": "cancelled"})
            continue
        spec = specs[step.view]
        emit(Event.VIEW_STARTED, {"view": step.view, "model_id": spec.model_id, "key": step.key,
                                  "evict_before": step.evict_before, "reuse_loaded": step.reuse_loaded})
        for victim in step.evict_before:
            if manager.unload(victim):
                emit(Event.STATUS, {"view": step.view, "stage": "evict", "message": f"Unloaded {victim.split('|')[0]}"})
        result = _run_view(manager, spec, step.view, prompts[step.view], system_prompt, params, stop_event, emit)
        results.append(result)
        emit(Event.VIEW_COMPLETE, result)
    return {"views": results, "plan": plan_data, "total_ms": (time.perf_counter() - started) * 1000.0}


def _run_view(manager: ModelManager, spec: ModelSpec, view: int, prompt: str, system_prompt: str | None,
              params: GenerationParams, stop_event: threading.Event, emit: Emit) -> dict[str, Any]:
    device = manager.resolve_device(spec)
    was_loaded = manager.get(spec.key(device)) is not None
    try:
        loaded = manager.ensure(spec, on_status=lambda msg: emit(Event.STATUS, {"view": view, "stage": "load",
                                                                               "message": msg}))
        engine = loaded.engine
        if not hasattr(engine, "stream_chat"):
            raise BadRequest(f"{spec.model_id} is not a text-generation model.")
        allocated_before = _allocated(device)
        result = engine.stream_chat([{"role": "user", "content": prompt}], params, stop_event=stop_event,
                                    on_text=lambda text: emit(Event.TOKEN_STREAM, {"view": view, "text": text}),
                                    system_prompt=system_prompt)
        metrics = result.metrics()
        if result.peak_memory_bytes is not None and loaded.engine_name != "mlx":
            metrics["model_peak_bytes"] = int(loaded.memory_bytes + max(result.peak_memory_bytes - allocated_before, 0))
        else:
            metrics["model_peak_bytes"] = result.peak_memory_bytes or loaded.memory_bytes
        metrics.update(load_ms=0.0 if was_loaded else loaded.load_ms, precision=spec.precision,
                       engine=loaded.engine_name, memory_bytes=loaded.memory_bytes, device=loaded.device)
        status = "cancelled" if result.finish_reason == "cancelled" else "finished"
        return {"view": view, "model_id": spec.model_id, "status": status, "text": result.text, "metrics": metrics}
    except Exception as exc:
        oom = devices.is_oom(exc)
        if oom:
            manager.unload_matching(spec.model_id)
            devices.empty_cache(device)
        code = ErrorCode.OOM if oom else getattr(exc, "code", ErrorCode.INTERNAL)
        hint = getattr(exc, "hint", None) if isinstance(exc, NoelleRuntimeError) else (
            "Out of memory: try a 4-bit build or fewer views." if oom else None)
        return {"view": view, "model_id": spec.model_id, "status": "error",
                "error": {"code": str(code), "message": getattr(exc, "message", str(exc)), "hint": hint}}


def _allocated(device: str) -> int:
    return int(devices.memory_snapshot(device).get("allocated") or 0)
