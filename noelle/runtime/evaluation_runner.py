"""Evaluation Lab executor (worker side): every case for one model, then the next model.

Model-major order loads each model once. Before the run, the combined footprint of
the selected models is planned with :func:`noelle.runtime.preflight.plan_queue`
(retain them all, or hot-swap: unload the previous model before loading the next).
Each case is generated greedily by default (temperature 0) so runs are repeatable,
scored locally with :mod:`noelle.runtime.evaluation`, and streamed back as a
``PROGRESS`` event; the terminal ``GENERATION_COMPLETE`` carries every result.
"""

from __future__ import annotations

import threading
import time
from dataclasses import replace
from typing import Any, Callable

from noelle.runtime import devices, evaluation
from noelle.runtime.errors import BadRequest, NoelleRuntimeError
from noelle.runtime.model_manager import ModelManager, ModelSpec
from noelle.runtime.preflight import plan_queue
from noelle.runtime.protocol import ErrorCode, Event, GenerationParams

Emit = Callable[[Event, dict[str, Any]], None]
MAX_MODELS = 8
RESULT_OUTPUT_CHARS = 4000


def run_evaluation(manager: ModelManager, payload: dict[str, Any], *, stop_event: threading.Event,
                   emit: Emit) -> dict[str, Any]:
    models = payload.get("models") or []
    if not models:
        raise BadRequest("Select at least one model.")
    if len(models) > MAX_MODELS:
        raise BadRequest(f"At most {MAX_MODELS} models per evaluation.")
    raw_cases = payload.get("cases") or []
    if not raw_cases:
        raise BadRequest("The dataset has no cases.")
    if len(raw_cases) > evaluation.MAX_CASES:
        raise BadRequest(f"At most {evaluation.MAX_CASES} cases per evaluation.")
    try:
        cases = [evaluation.EvalCase.from_dict(case, index) for index, case in enumerate(raw_cases)]
    except evaluation.DatasetError as exc:
        raise BadRequest(f"Invalid dataset: {exc}") from exc
    params = GenerationParams.from_dict({"temperature": 0.0, **(payload.get("params") or {})})
    system_prompt = payload.get("system_prompt") or None

    specs: list[ModelSpec] = []
    jobs: list[tuple[int, str, int]] = []
    resident = manager.resident_sizes()
    for index, entry in enumerate(models, start=1):
        spec = ModelSpec.from_payload({**entry, "task": entry.get("task") or "text-generation"})
        specs.append(spec)
        device = manager.resolve_device(spec)
        key = spec.key(device)
        size = resident.get(key)
        if size is None:
            size = manager.estimate_bytes(spec, manager.resolve_source(spec), device)
        jobs.append((index, key, size))
    plan = plan_queue(jobs, manager.budget().usable_bytes, resident=resident)
    plan_data = {**plan.to_dict(), "device": manager.device}
    emit(Event.QUEUE_PLAN, plan_data)

    total = len(cases) * len(specs)
    progress = {"done": 0}
    results: list[dict[str, Any]] = []
    started = time.perf_counter()
    for step in plan.steps:
        spec = specs[step.view - 1]
        if stop_event.is_set():
            results.append(_model_result(step.view, spec, "cancelled", [], {}))
            continue
        emit(Event.VIEW_STARTED, {"view": step.view, "model_id": spec.model_id, "key": step.key,
                                  "evict_before": step.evict_before, "reuse_loaded": step.reuse_loaded})
        for victim in step.evict_before:
            if manager.unload(victim):
                emit(Event.STATUS, {"view": step.view, "stage": "evict", "message": f"Unloaded {victim.split('|')[0]}"})
        result = _run_model(manager, spec, step.view, cases, params, system_prompt, stop_event, emit, progress, total)
        results.append(result)
        emit(Event.VIEW_COMPLETE, {k: v for k, v in result.items() if k != "cases"})
    return {"models": results, "plan": plan_data, "cases": len(cases),
            "total_ms": (time.perf_counter() - started) * 1000.0, "params": params.to_dict()}


def _run_model(manager: ModelManager, spec: ModelSpec, view: int, cases: list[evaluation.EvalCase],
               params: GenerationParams, system_prompt: str | None, stop_event: threading.Event, emit: Emit,
               progress: dict[str, int], total: int) -> dict[str, Any]:
    device = manager.resolve_device(spec)
    was_loaded = manager.get(spec.key(device)) is not None
    try:
        loaded = manager.ensure(spec, on_status=lambda msg: emit(Event.STATUS, {"view": view, "stage": "load",
                                                                               "message": msg}))
        if not hasattr(loaded.engine, "stream_chat"):
            raise BadRequest(f"{spec.model_id} is not a text-generation model.")
    except Exception as exc:
        progress["done"] += len(cases)
        emit(Event.PROGRESS, {"current": progress["done"], "total": total, "view": view, "model_id": spec.model_id})
        return _model_result(view, spec, "error", [], {}, error=_error(manager, spec, device, exc))

    metrics: dict[str, Any] = {"engine": loaded.engine_name, "device": loaded.device, "precision": spec.precision,
                               "memory_bytes": loaded.memory_bytes, "load_ms": 0.0 if was_loaded else loaded.load_ms}
    allocated_before = int(devices.memory_snapshot(device).get("allocated") or 0)
    rows: list[dict[str, Any]] = []
    peak = 0
    error: dict[str, Any] | None = None
    with manager.pinned(loaded.key):
        for case in cases:
            if stop_event.is_set():
                break
            case_params = replace(params, max_new_tokens=case.max_new_tokens) if case.max_new_tokens else params
            try:
                generated = loaded.engine.stream_chat(case.chat_messages(), case_params, stop_event=stop_event,
                                                      on_text=lambda _text: None,
                                                      system_prompt=case.system or system_prompt)
            except Exception as exc:
                row = _row(view, case, error=str(exc) or type(exc).__name__)
                if devices.is_oom(exc):
                    error = _error(manager, spec, device, exc)
                    rows.append(row)
                    progress["done"] += 1
                    break
            else:
                peak = max(peak, int(generated.peak_memory_bytes or 0))
                score = evaluation.score_case(case, generated.text)
                row = _row(view, case, output=generated.text, score=score, ttft_ms=generated.ttft_ms,
                           tokens_per_second=generated.tokens_per_second,
                           completion_tokens=generated.completion_tokens, prompt_tokens=generated.prompt_tokens,
                           latency_ms=generated.total_ms, finish_reason=generated.finish_reason)
            rows.append(row)
            progress["done"] += 1
            emit(Event.PROGRESS, {"current": progress["done"], "total": total, "view": view,
                                  "model_id": spec.model_id, "case": row})
    manager.touch(loaded.key)
    if peak and loaded.engine_name != "mlx":
        metrics["peak_bytes"] = int(loaded.memory_bytes + max(peak - allocated_before, 0))
    else:
        metrics["peak_bytes"] = peak or loaded.memory_bytes
    skipped = len(cases) - len(rows)
    if skipped:
        progress["done"] += skipped
        emit(Event.PROGRESS, {"current": progress["done"], "total": total, "view": view, "model_id": spec.model_id})
    status = "error" if error else "cancelled" if stop_event.is_set() and skipped else "finished"
    return _model_result(view, spec, status, rows, metrics, error=error)


def _row(view: int, case: evaluation.EvalCase, *, output: str = "", score: evaluation.CaseScore | None = None,
         error: str | None = None, **metrics: Any) -> dict[str, Any]:
    return {
        "view": view,
        "case_id": case.id,
        "prompt": case.display_prompt()[:2000],
        "expected": case.expected,
        "output": output[:RESULT_OUTPUT_CHARS],
        "score": score.score if score else None,
        "passed": score.passed if score else None,
        "detail": score.detail if score else (error or ""),
        "tags": case.tags,
        "error": error,
        **metrics,
    }


def _model_result(view: int, spec: ModelSpec, status: str, rows: list[dict[str, Any]], metrics: dict[str, Any],
                  error: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"view": view, "model_id": spec.model_id, "precision": spec.precision, "status": status,
            "summary": evaluation.summarize(rows), "metrics": metrics, "error": error, "cases": rows}


def _error(manager: ModelManager, spec: ModelSpec, device: str, exc: BaseException) -> dict[str, Any]:
    oom = devices.is_oom(exc)
    if oom:
        manager.unload_matching(spec.model_id)
        devices.empty_cache(device)
    code = ErrorCode.OOM if oom else getattr(exc, "code", ErrorCode.INTERNAL)
    hint = getattr(exc, "hint", None) if isinstance(exc, NoelleRuntimeError) else (
        "Out of memory: try a 4-bit build or fewer models." if oom else None)
    return {"code": str(code), "message": getattr(exc, "message", None) or str(exc) or type(exc).__name__,
            "hint": hint}
