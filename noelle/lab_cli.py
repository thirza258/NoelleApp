"""Command-line parity for the Evaluation Lab and the model analyzer.

``noelle eval``     run datasets against local models in a private worker; thresholds and a
                    baseline make it a CI regression gate (exit code 1 when a check fails)
``noelle analyze``  the Workspace "Analysis" for any Hub repository, from the terminal

Recipe files (YAML or JSON) bundle a whole run::

    name: nightly
    models: [HuggingFaceTB/SmolLM2-360M-Instruct, local/qwen-mlx]
    datasets: [builtin:reasoning-math, ./my-cases.jsonl]
    params: {max_new_tokens: 256, temperature: 0}
    thresholds: [accuracy>=0.6, tok_s>=20, ttft_ms<=1500]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

EXIT_OK, EXIT_FAILED_CHECKS, EXIT_ERROR = 0, 1, 2


def add_eval_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", "-m", action="append", default=[],
                        help="model to evaluate: API name, repo id or local build (repeatable)")
    parser.add_argument("--dataset", "-d", action="append", default=[],
                        help="builtin:<name> or a .jsonl file (repeatable)")
    parser.add_argument("--recipe", default=None, help="YAML/JSON recipe with models, datasets, params, thresholds")
    parser.add_argument("--precision", default=None, help="precision for Hub checkpoints (auto, fp16, int8, nf4…)")
    parser.add_argument("--device", default=None, help="auto, cuda, mps or cpu")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None, help="default 0 (greedy, repeatable)")
    parser.add_argument("--system", default=None, help="system prompt for every case")
    parser.add_argument("--threshold", "-t", action="append", default=[],
                        help="e.g. accuracy>=0.8, tok_s>=20, ttft_ms<=1000 (repeatable)")
    parser.add_argument("--baseline", default=None,
                        help="JSON written by a previous --json run; fail when accuracy or speed regresses")
    parser.add_argument("--max-regression", type=float, default=0.10,
                        help="allowed relative drop against --baseline (default 0.10 = 10%%)")
    parser.add_argument("--json", dest="json_out", default=None, help="write the experiments as JSON")
    parser.add_argument("--markdown", default=None, help="write a Markdown report")
    parser.add_argument("--no-save", action="store_true", help="do not store the run in Noelle's experiment history")
    parser.add_argument("--list-datasets", action="store_true", help="list built-in datasets and exit")


def add_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", "-m", required=True, help="API name, repo id or local build")
    parser.add_argument("prompt", nargs="?", default=None, help="prompt text (default: read stdin)")
    parser.add_argument("--system", default=None)
    parser.add_argument("--precision", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--metrics", action="store_true", help="print TTFT, speed and memory to stderr")


def add_analyze_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("repo_id", help="Hugging Face repository, e.g. Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--profile", default="balanced",
                        choices=["balanced", "quality", "fast", "low_memory", "max_context"])
    parser.add_argument("--json", action="store_true", help="print the full analysis as JSON")


# ---------------------------------------------------------------------------
# eval
# ---------------------------------------------------------------------------


def load_recipe(path: str) -> dict[str, Any]:
    text = Path(path).expanduser().read_text(encoding="utf-8")
    if path.endswith((".yaml", ".yml")):
        import yaml

        data = yaml.safe_load(text) or {}
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("a recipe must be a mapping")
    return data


def _threshold_strings(value: Any) -> list[str]:
    if isinstance(value, dict):  # {accuracy: ">=0.8"}
        return [f"{k}{v}" if str(v)[:1] in "<>=" else f"{k}>={v}" for k, v in value.items()]
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    return [str(value)] if value else []


def _resolve_models(names: list[str], precision: str | None) -> tuple[list[dict[str, Any]], list[Any]]:
    from noelle.app.services.model_registry import get_registry, remote_code_trusted
    from noelle.config import get_settings
    from noelle.runtime.hub_cache import cached_snapshot_path

    registry = get_registry()
    payloads, records = [], []
    for name in names:
        record = registry.find_by_api_name(name) or registry.get_model(name)
        if record is not None and record.exists_on_disk():
            payload = record.worker_payload(get_settings())
            if precision and record.source == "hub" and record.precision == "auto":
                payload["precision"] = precision
        elif cached_snapshot_path(name) is not None:
            payload = {"model_id": name, "task": "text-generation", "precision": precision or "auto",
                       "trust_remote_code": remote_code_trusted(get_settings(), name)}
        else:
            raise ValueError(f"'{name}' is not downloaded. Download it in Noelle first (or with `noelle download`).")
        payloads.append(payload)
        records.append(record)
    return payloads, records


def run_eval(args: argparse.Namespace) -> int:
    from noelle.runtime import evaluation

    if args.list_datasets:
        for item in evaluation.builtin_catalog():
            print(f"builtin:{item['name']:<24} {item['cases']:>3} cases  {item['category']:<18} {item['title']}")
        return EXIT_OK
    try:
        recipe = load_recipe(args.recipe) if args.recipe else {}
        models = list(recipe.get("models") or []) + list(args.model)
        dataset_refs = list(recipe.get("datasets") or ([recipe["dataset"]] if recipe.get("dataset") else []))
        dataset_refs += list(args.dataset)
        if not models or not dataset_refs:
            print("Give at least one --model and one --dataset (or a --recipe).", file=sys.stderr)
            return EXIT_ERROR
        thresholds = [evaluation.parse_threshold(t)
                      for t in _threshold_strings(recipe.get("thresholds")) + list(args.threshold)]
        datasets = [evaluation.load_dataset(ref) for ref in dataset_refs]
        params: dict[str, Any] = {"temperature": 0.0, **(recipe.get("params") or {})}
        if args.max_new_tokens is not None:
            params["max_new_tokens"] = args.max_new_tokens
        if args.temperature is not None:
            params["temperature"] = args.temperature
        payloads, records = _resolve_models(models, args.precision or recipe.get("precision"))
        baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8")) if args.baseline else None
    except (OSError, ValueError, evaluation.DatasetError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR

    from noelle.app.services import experiments
    from noelle.app.services.hardware_manager import probe_profile
    from noelle.app.services.model_registry import get_registry
    from noelle.runtime.client import WorkerError, spawn_worker
    from noelle.runtime.protocol import Command, Event

    system_prompt = args.system or recipe.get("system_prompt")
    hardware = experiments.hardware_snapshot(probe_profile())
    print(f"Starting the model runtime on {hardware.get('name') or hardware.get('device')}…", file=sys.stderr)
    worker = spawn_worker(parent_pid=os.getpid(), device=args.device or recipe.get("device"))
    stored: list[dict[str, Any]] = []
    failures: list[str] = []
    try:
        client = worker.client()
        for dataset in datasets:
            print(f"\n== {dataset.title} ({len(dataset.cases)} cases, {dataset.source})", file=sys.stderr)
            payload = {"models": payloads, "cases": [c.to_dict() for c in dataset.cases], "params": params,
                       "system_prompt": system_prompt}
            final: dict[str, Any] | None = None
            for message in client.stream(Command.RUN_EVAL, payload, timeout=None):
                event, data = message["event"], message["payload"]
                if event == Event.STATUS and data.get("stage") == "load":
                    print(f"   {data.get('message')}", file=sys.stderr)
                elif event == Event.PROGRESS and data.get("case"):
                    _print_case(data)
                elif event == Event.GENERATION_COMPLETE:
                    final = data
                elif event == Event.ERROR:
                    raise WorkerError(data)
            if final is None:
                raise WorkerError({"code": "INTERNAL", "message": "the runtime ended the evaluation without a result"})
            experiment_id = None
            if not args.no_save:
                experiment_id = experiments.record_evaluation(
                    get_registry(), final, dataset=dataset.meta(), params=params, hardware=hardware,
                    models=payloads, records=records, kind="cli", name=recipe.get("name"))
            experiment = get_registry().get_experiment(experiment_id) if experiment_id else _unsaved(
                final, dataset, params, hardware, payloads, records)
            stored.append(experiment)
            _print_leaderboard(experiment)
            failures += _check(experiment, thresholds, baseline, args.max_regression)
    except WorkerError as exc:
        print(f"error: {exc} {exc.hint or ''}".strip(), file=sys.stderr)
        return EXIT_ERROR
    except KeyboardInterrupt:
        print("cancelled", file=sys.stderr)
        return EXIT_ERROR
    finally:
        worker.stop()
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(stored, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    if args.markdown:
        Path(args.markdown).write_text("\n".join(experiments.export_markdown(e) for e in stored), encoding="utf-8")
    if failures:
        print("\nFAILED checks:", file=sys.stderr)
        for failure in failures:
            print(f"  ✗ {failure}", file=sys.stderr)
        return EXIT_FAILED_CHECKS
    if thresholds or baseline:
        print("\nAll checks passed.", file=sys.stderr)
    return EXIT_OK


def _unsaved(final: dict[str, Any], dataset: Any, params: dict[str, Any], hardware: dict[str, Any],
             payloads: list[dict[str, Any]], records: list[Any]) -> dict[str, Any]:
    from noelle.app.services import experiments
    from noelle.app.services.model_registry import ModelRegistry

    scratch = ModelRegistry(":memory:")
    experiment_id = experiments.record_evaluation(scratch, final, dataset=dataset.meta(), params=params,
                                                  hardware=hardware, models=payloads, records=records, kind="cli")
    return scratch.get_experiment(experiment_id) or {}


def _print_case(data: dict[str, Any]) -> None:
    case = data["case"]
    mark = "·" if case.get("passed") is None else ("✓" if case.get("passed") else "✗")
    if case.get("error"):
        mark = "!"
    speed = f"{case['tokens_per_second']:.1f} tok/s" if case.get("tokens_per_second") else ""
    print(f"   [{data['current']}/{data['total']}] {mark} {data.get('model_id')} · {case.get('case_id')} {speed}",
          file=sys.stderr)


def _print_leaderboard(experiment: dict[str, Any]) -> None:
    from noelle.app.services.experiments import leaderboard

    rows = leaderboard(experiment)
    dataset = (experiment.get("dataset") or {}).get("name")
    print(f"\n{dataset}  (experiment {experiment.get('id', 'unsaved')})")
    print(f"{'model':<44} {'accuracy':>9} {'tok/s':>8} {'TTFT ms':>9} {'status':>10}")
    for row in rows:
        accuracy = "—" if row["accuracy"] is None else f"{row['accuracy'] * 100:.1f}%"
        tps = "—" if not row["tokens_per_second"] else f"{row['tokens_per_second']:.1f}"
        ttft = "—" if row["ttft_ms"] is None else f"{row['ttft_ms']:.0f}"
        print(f"{str(row['model'])[:44]:<44} {accuracy:>9} {tps:>8} {ttft:>9} {row['status'] or '':>10}")


def _check(experiment: dict[str, Any], thresholds: list[Any], baseline: Any, max_regression: float) -> list[str]:
    from noelle.runtime import evaluation

    failures = []
    dataset = (experiment.get("dataset") or {}).get("name")
    for run in experiment.get("runs") or []:
        name = run.get("api_name") or run.get("model_id")
        summary = run.get("summary") or {}
        if run.get("status") != "finished":
            failures.append(f"{name} on {dataset}: run {run.get('status')} "
                            f"({(run.get('error') or {}).get('message', '')})".strip())
            continue
        for message in evaluation.check_thresholds(summary, thresholds):
            failures.append(f"{name} on {dataset}: {message}")
        previous = _baseline_summary(baseline, dataset, name)
        if previous:
            for metric, higher_is_better in (("accuracy", True), ("tokens_per_second", True), ("ttft_ms_p50", False)):
                old, new = previous.get(metric), summary.get(metric)
                if old is None or new is None or old == 0:
                    continue
                change = (new - old) / abs(old)
                if (higher_is_better and change < -max_regression) or (not higher_is_better and change > max_regression):
                    failures.append(f"{name} on {dataset}: {metric} regressed {old:.4g} → {new:.4g} "
                                    f"({change * 100:+.1f}%, allowed {max_regression * 100:.0f}%)")
    return failures


def _baseline_summary(baseline: Any, dataset: str | None, model: str | None) -> dict[str, Any] | None:
    for experiment in baseline or []:
        if (experiment.get("dataset") or {}).get("name") != dataset:
            continue
        for run in experiment.get("runs") or []:
            if (run.get("api_name") or run.get("model_id")) == model and run.get("status") == "finished":
                return run.get("summary") or {}
    return None


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


def run_prompt(args: argparse.Namespace) -> int:
    """One prompt, streamed to stdout, in a private worker (``noelle run -m MODEL "prompt"``)."""
    prompt = args.prompt if args.prompt is not None else sys.stdin.read()
    if not prompt.strip():
        print("error: empty prompt", file=sys.stderr)
        return EXIT_ERROR
    try:
        (payload,), _records = _resolve_models([args.model], args.precision)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    from noelle.runtime.client import spawn_worker
    from noelle.runtime.protocol import Command, Event

    worker = spawn_worker(parent_pid=os.getpid(), device=args.device)
    try:
        request = {**payload, "messages": [{"role": "user", "content": prompt}], "system_prompt": args.system,
                   "params": {"max_new_tokens": args.max_new_tokens, "temperature": args.temperature}}
        for message in worker.client().stream(Command.GENERATE_STREAM, request, timeout=None):
            event, data = message["event"], message["payload"]
            if event == Event.TOKEN_STREAM:
                sys.stdout.write(data.get("text", ""))
                sys.stdout.flush()
            elif event == Event.STATUS and data.get("stage") == "load":
                print(data.get("message", ""), file=sys.stderr)
            elif event == Event.ERROR:
                print(f"\nerror: {data.get('message')} {data.get('hint') or ''}".rstrip(), file=sys.stderr)
                return EXIT_ERROR
            elif event == Event.GENERATION_COMPLETE:
                sys.stdout.write("\n")
                sys.stdout.flush()
                metrics = data.get("metrics") or {}
                if args.metrics:
                    print(f"ttft {metrics.get('ttft_ms') or 0:.0f} ms · {metrics.get('tokens_per_second') or 0:.1f} "
                          f"tok/s · {metrics.get('completion_tokens')} tokens · {metrics.get('engine')} on "
                          f"{metrics.get('device')}", file=sys.stderr)
    except KeyboardInterrupt:
        return EXIT_ERROR
    finally:
        worker.stop()
    return EXIT_OK


# ---------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------


def run_analyze(args: argparse.Namespace) -> int:
    from noelle.app.services.hardware_manager import probe_profile
    from noelle.app.services.hf_client import HFClient
    from noelle.app.services.model_registry import get_registry
    from noelle.config import get_settings
    from noelle.runtime import advisor, capabilities
    from noelle.runtime.preflight import format_bytes

    client = HFClient(get_settings().get("hub.endpoint"))
    try:
        summary = client.model_info(args.repo_id)
    except Exception as exc:  # network, 404, gated
        print(f"error: could not read {args.repo_id} from the Hub: {exc}", file=sys.stderr)
        return EXIT_ERROR
    config = client.fetch_config(args.repo_id)
    tokenizer = None
    text = client.fetch_file_text(args.repo_id, "tokenizer_config.json", max_bytes=1_000_000)
    if text:
        try:
            tokenizer = json.loads(text)
        except ValueError:
            tokenizer = None
    registry = get_registry()
    cap = capabilities.analyze(summary, config or None, tokenizer_config=tokenizer)
    local = {r.precision: r.repo_id for r in registry.list_models()
             if r.base_model == args.repo_id and r.source == "quantized" and r.exists_on_disk()}
    assessment = advisor.assess(cap, probe_profile(), config=config or None, profile=args.profile,
                                measured=advisor.summarize_measurements(registry.measurements(args.repo_id)),
                                local_variants=local)
    if args.json:
        print(json.dumps({"capability": cap.to_dict(), "assessment": assessment.to_dict()}, indent=2, default=str))
        return EXIT_OK
    hw = assessment.hardware
    print(f"{cap.model_id}  —  {assessment.score}/100 {assessment.grade} ({assessment.compatibility.label})")
    print(f"on {hw.name} · {format_bytes(hw.usable_bytes)} usable · profile {args.profile}\n")
    for item in assessment.items:
        print(f"  {item.label:<15} {item.points:>2}/{item.max_points:<3} {item.note}")
    if assessment.cap_note:
        print(f"  {assessment.cap_note}")
    rec = assessment.recommended
    print(f"\nRecommended: {rec.label + ' via ' + rec.runtime if rec else 'nothing fits on this machine'}")
    print(f"Speed:       {assessment.speed_text()}")
    if assessment.context_tokens:
        print(f"Context:     up to {assessment.context_tokens:,} tokens fit")
    print("\nOptions:")
    for option in assessment.options:
        fits = "—" if option.fits is None else ("fits" if option.fits else "too big")
        memory = format_bytes(option.memory_bytes) if option.memory_bytes else "?"
        print(f"  {option.label:<24} {option.runtime:<13} {memory:>9}  {fits:<8} {option.status:<12} "
              f"{option.estimate_text() or ''}")
    for note in assessment.notes:
        print(f"• {note}")
    return EXIT_OK
