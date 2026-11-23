"""Experiment tracking for the Evaluation Lab and the Compare page (Qt-free; used by the UI and the CLI).

An experiment stores what is needed to reproduce and trust a result:

* the dataset (name, source, SHA-256 digest of its cases),
* the generation config (temperature, max tokens, …),
* the hardware and software it ran on (device, memory, OS, Python, Noelle, engines),
* per model: repo, revision / commit hash, local path, precision, engine, device and
  every case's output, score and timing.
"""

from __future__ import annotations

import json
import platform
import time
from typing import Any, Iterable

from noelle import __version__
from noelle.app.services.model_registry import ModelRecord, ModelRegistry
from noelle.runtime.preflight import format_bytes


def provenance(record: ModelRecord | None, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    """Where a model's weights came from (§ model provenance)."""
    data: dict[str, Any] = {"model_id": (payload or {}).get("model_id")}
    if record is not None:
        data.update(repo_id=record.repo_id, source=record.source, revision=record.revision,
                    commit_hash=record.commit_hash, local_path=record.local_path, size_bytes=record.size_bytes,
                    license=record.license, base_model=record.base_model, gguf_file=record.gguf_file,
                    downloaded_at=record.downloaded_at, trust_remote_code=record.trust_remote_code,
                    publisher=record.repo_id.split("/")[0] if "/" in record.repo_id else None)
    return {k: v for k, v in data.items() if v is not None}


def hardware_snapshot(profile: Any = None, heartbeat: dict[str, Any] | None = None) -> dict[str, Any]:
    beat = heartbeat or {}
    data: dict[str, Any] = {
        "os": f"{platform.system()} {platform.release()}",
        "machine": platform.machine(),
        "python": platform.python_version(),
        "noelle": __version__,
    }
    if profile is not None:
        data.update(device=profile.device, name=profile.name, usable_bytes=profile.usable_bytes,
                    total_bytes=profile.budget.total_bytes, system_ram_bytes=profile.budget.system_ram_bytes,
                    backends=sorted(k for k, ok in profile.backends.items() if ok))
    if beat.get("device"):
        data["worker_device"] = beat.get("device")
    return data


def record_evaluation(registry: ModelRegistry, result: dict[str, Any], *, dataset: dict[str, Any],
                      params: dict[str, Any], hardware: dict[str, Any], models: list[dict[str, Any]],
                      records: list[ModelRecord | None], kind: str = "evaluation", name: str | None = None,
                      status: str | None = None) -> str:
    """Persist a ``RUN_EVAL`` result. ``models``/``records`` follow the request order (view = index + 1)."""
    title = name or f"{dataset.get('title') or dataset.get('name') or 'Evaluation'} · {time.strftime('%Y-%m-%d %H:%M')}"
    experiment_id = registry.create_experiment(kind=kind, name=title, dataset=dataset, params=params,
                                               hardware=hardware)
    best: tuple[float, str] | None = None
    statuses = []
    for run in result.get("models") or []:
        index = int(run.get("view") or 1) - 1
        payload = models[index] if 0 <= index < len(models) else {}
        record = records[index] if 0 <= index < len(records) else None
        api_name = record.api_name if record is not None else run.get("model_id")
        run_id = registry.add_experiment_run(experiment_id, {**run, "api_name": api_name,
                                                             "provenance": provenance(record, payload)})
        registry.add_experiment_cases(experiment_id, run_id, run.get("cases") or [])
        statuses.append(run.get("status"))
        accuracy = (run.get("summary") or {}).get("accuracy")
        if accuracy is not None and (best is None or accuracy > best[0]):
            best = (accuracy, api_name)
    final = status or ("finished" if statuses and all(s == "finished" for s in statuses) else
                       "cancelled" if "cancelled" in statuses else "partial" if "finished" in statuses else "failed")
    registry.finish_experiment(experiment_id, final, {
        "models": len(statuses), "best_accuracy": best[0] if best else None, "best_model": best[1] if best else None,
        "total_ms": result.get("total_ms"), "cases": result.get("cases")})
    return experiment_id


def record_compare(registry: ModelRegistry, *, prompt: str, system_prompt: str | None, params: dict[str, Any],
                   results: Iterable[dict[str, Any]], records: dict[int, ModelRecord | None],
                   hardware: dict[str, Any]) -> str:
    """Store a Compare run as an experiment with one ad-hoc case per view."""
    dataset = {"name": "compare-prompt", "title": "Compare prompt", "category": "ad-hoc", "cases": 1,
               "source": "compare"}
    converted = []
    for result in results:
        metrics = dict(result.get("metrics") or {})
        metrics.setdefault("peak_bytes", metrics.get("model_peak_bytes"))
        case = {"case_id": "prompt", "prompt": prompt, "output": result.get("text", ""),
                "ttft_ms": metrics.get("ttft_ms"), "tokens_per_second": metrics.get("tokens_per_second"),
                "completion_tokens": metrics.get("completion_tokens"), "latency_ms": metrics.get("total_ms"),
                "error": (result.get("error") or {}).get("message") if result.get("error") else None}
        summary = {"cases": 1, "scored": 0, "accuracy": None, "ttft_ms_p50": metrics.get("ttft_ms"),
                   "tokens_per_second": metrics.get("tokens_per_second")}
        converted.append({"view": result.get("view"), "model_id": result.get("model_id"),
                          "precision": metrics.get("precision"), "status": result.get("status"), "summary": summary,
                          "metrics": metrics, "error": result.get("error"), "cases": [case]})
    views = sorted(records)
    models = [{"model_id": (records[v].repo_id if records[v] else None)} for v in views]
    ordered = [records[v] for v in views]
    remapped = []
    for run in converted:
        try:
            position = views.index(int(run.get("view") or 0))
        except ValueError:
            continue
        remapped.append({**run, "view": position + 1})
    return record_evaluation(registry, {"models": remapped, "cases": 1}, dataset=dataset,
                             params={**params, "system_prompt": system_prompt}, hardware=hardware, models=models,
                             records=ordered, kind="compare", name=f"Compare · {prompt[:48]}")


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------


def export_json(experiment: dict[str, Any]) -> str:
    return json.dumps(experiment, indent=2, ensure_ascii=False, default=str)


def leaderboard(experiment: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for run in experiment.get("runs") or []:
        summary = run.get("summary") or {}
        metrics = run.get("metrics") or {}
        rows.append({
            "model": run.get("api_name") or run.get("model_id"),
            "precision": run.get("precision"),
            "engine": metrics.get("engine"),
            "device": metrics.get("device"),
            "status": run.get("status"),
            "accuracy": summary.get("accuracy"),
            "mean_score": summary.get("mean_score"),
            "ttft_ms": summary.get("ttft_ms_p50"),
            "tokens_per_second": summary.get("tokens_per_second"),
            "peak_memory_bytes": metrics.get("peak_bytes"),
            "load_ms": metrics.get("load_ms"),
            "errors": summary.get("errors"),
        })
    rows.sort(key=lambda r: (-(r["accuracy"] if r["accuracy"] is not None else -1),
                             -(r["tokens_per_second"] or 0)))
    return rows


def export_markdown(experiment: dict[str, Any]) -> str:
    dataset = experiment.get("dataset") or {}
    hardware = experiment.get("hardware") or {}
    params = experiment.get("params") or {}
    created = time.strftime("%Y-%m-%d %H:%M", time.localtime(experiment.get("created_at") or time.time()))
    lines = [
        f"# {experiment.get('name') or 'Evaluation'}",
        "",
        f"- Experiment: `{experiment.get('id')}` ({experiment.get('kind')}, {experiment.get('status')}) — {created}",
        f"- Dataset: {dataset.get('title') or dataset.get('name')} · {dataset.get('cases')} cases · "
        f"`{dataset.get('source')}` · sha256 `{str(dataset.get('digest') or '')[:16]}`",
        f"- Hardware: {hardware.get('name') or hardware.get('device')} · {format_bytes(hardware.get('usable_bytes'))} "
        f"usable · {hardware.get('os')} · Python {hardware.get('python')} · Noelle {hardware.get('noelle')}",
        "- Generation: " + ", ".join(f"{k}={v}" for k, v in sorted(params.items()) if v not in (None, [], "")),
        "",
        "| Model | Precision | Engine | Accuracy | Mean score | TTFT p50 | tok/s | Peak memory | Status |",
        "|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in leaderboard(experiment):
        lines.append("| {model} | {precision} | {engine} | {acc} | {score} | {ttft} | {tps} | {peak} | {status} |".format(
            model=row["model"], precision=row["precision"] or "", engine=row["engine"] or "",
            acc=_pct(row["accuracy"]), score=_num(row["mean_score"], 2), ttft=_num(row["ttft_ms"], 0, " ms"),
            tps=_num(row["tokens_per_second"], 1), peak=format_bytes(row["peak_memory_bytes"]) if row[
                "peak_memory_bytes"] else "—", status=row["status"] or ""))
    lines.append("")
    lines.append("## Provenance")
    lines.append("")
    for run in experiment.get("runs") or []:
        prov = run.get("provenance") or {}
        commit = prov.get("commit_hash") or prov.get("revision") or "unknown revision"
        lines.append(f"- **{run.get('api_name') or run.get('model_id')}** — {prov.get('source', 'hub')} "
                     f"`{prov.get('repo_id', run.get('model_id'))}` @ `{commit}`"
                     + (f" (base {prov['base_model']})" if prov.get("base_model") else ""))
    failing = [(run, case) for run in experiment.get("runs") or [] for case in run.get("cases") or []
               if case.get("passed") is False or case.get("error")]
    if failing:
        lines += ["", "## Failed cases", "", "| Model | Case | Detail |", "|---|---|---|"]
        for run, case in failing[:200]:
            detail = (case.get("error") or case.get("detail") or "").replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {run.get('api_name') or run.get('model_id')} | {case.get('case_id')} | {detail[:160]} |")
    return "\n".join(lines) + "\n"


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.1f}%"


def _num(value: float | None, digits: int, suffix: str = "") -> str:
    return "—" if value is None else f"{value:.{digits}f}{suffix}"
