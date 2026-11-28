"""``noelle train`` CLI: prepare datasets, plan, run and inspect local fine-tuning."""

from __future__ import annotations

import argparse
import json
import signal
import sys
import threading
import time
from typing import Any

from noelle.training.config import ConfigError, TrainingConfig, estimate_plan
from noelle.training.dataset import analyze_dataset, prepare_dataset
from noelle.training.hardware import hardware_info
from noelle.training.store import TrainingStore


def add_training_arguments(parser: argparse.ArgumentParser) -> None:
    sub = parser.add_subparsers(dest="train_command")

    run = sub.add_parser("run", help="start a training run from a recipe (YAML or JSON)")
    run.add_argument("--config", required=True)
    run.add_argument("--json", action="store_true", help="emit newline-delimited JSON events")
    run.add_argument("--resume", default=None, help="resume from a checkpoint directory")

    prepare = sub.add_parser("prepare", help="normalize, split and report a dataset")
    prepare.add_argument("--dataset", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--max-seq-length", type=int, default=1024)
    prepare.add_argument("--validation-ratio", type=float, default=0.1)
    prepare.add_argument("--no-dedupe", action="store_true")
    prepare.add_argument("--trim-long", action="store_true")
    prepare.add_argument("--json", action="store_true")

    plan = sub.add_parser("plan", help="estimate memory, time and risk for a recipe")
    plan.add_argument("--config", required=True)
    plan.add_argument("--json", action="store_true")

    merge = sub.add_parser("merge", help="merge a LoRA adapter into a full checkpoint")
    merge.add_argument("--base", required=True)
    merge.add_argument("--adapter", required=True)
    merge.add_argument("--output", required=True)
    merge.add_argument("--trust-remote-code", action="store_true")

    listing = sub.add_parser("list", help="list recent training runs")
    listing.add_argument("--limit", type=int, default=20)
    listing.add_argument("--json", action="store_true")

    status = sub.add_parser("status", help="show one training run")
    status.add_argument("run_id")
    status.add_argument("--json", action="store_true")

    doctor = sub.add_parser("doctor", help="check that the training packages are installed")
    doctor.add_argument("--json", action="store_true")


def _human_event(event: dict[str, Any]) -> str | None:
    kind = event.get("type")
    if kind == "training.progress":
        loss = event.get("loss")
        return (f"step {event.get('step') or 0}/{event.get('total_steps') or '?'}"
                + (f"  loss {loss:.4f}" if isinstance(loss, (int, float)) else ""))
    if kind == "training.checkpoint":
        return f"checkpoint saved: {event.get('path')}"
    if kind == "training.dataset":
        report = event.get("report") or {}
        return f"dataset: {report.get('train', 0)} train / {report.get('validation', 0)} validation rows"
    if kind == "training.error":
        return f"error: {event.get('message')}"
    if kind == "training.completed":
        return f"completed: {event.get('output_path')}"
    if kind == "training.warning":
        return f"warning: {event.get('message')}"
    return None


def _print_plan(plan: Any) -> None:
    print(f"Mode: {plan.mode} ({plan.precision})")
    print(f"Risk: {plan.risk}")
    print(f"Effective batch: {plan.effective_batch} · expected steps: {plan.expected_steps}")
    if plan.estimated_vram_bytes:
        print(f"Estimated VRAM: {plan.estimated_vram_bytes / 1e9:.1f} GB")
    print(f"Estimated RAM: {plan.estimated_ram_bytes / 1e9:.1f} GB")
    print(f"Estimated disk: {plan.disk_bytes / 1e9:.1f} GB")
    print(f"Estimated time: {plan.estimated_seconds / 60:.0f} min")
    for note in plan.notes:
        print(f"  note: {note}")


def _run(args: argparse.Namespace) -> int:
    try:
        config = TrainingConfig.from_file(args.config)
        config.validate()
    except Exception as exc:  # noqa: BLE001 - hand-edited recipes are user input
        print(f"Invalid training configuration: {exc}", file=sys.stderr)
        return 2
    if args.resume:
        config.resume_from = str(args.resume)
    store = TrainingStore()
    run = store.create(config)
    run_id = run["id"]
    store.config_path(run_id).write_text(json.dumps(config.to_dict(), indent=2, default=str), encoding="utf-8")
    print(f"Training run {run_id}", file=sys.stderr if args.json else sys.stdout)
    stop_event = threading.Event()

    def request_stop(_signum: int, _frame: Any) -> None:
        stop_event.set()
        print("Stopping after the current step…", file=sys.stderr)

    try:
        signal.signal(signal.SIGINT, request_stop)
        signal.signal(signal.SIGTERM, request_stop)
    except ValueError:  # not the main thread (embedded use)
        pass

    def emit(event: dict[str, Any]) -> None:
        if args.json:
            print(json.dumps(event, default=str), flush=True)
            return
        text = _human_event(event)
        if text:
            print(text, flush=True)

    from noelle.training.trainer import TrainingDependencyError, run_training

    started = time.monotonic()
    try:
        manifest = run_training(config, run_id, store, emit, stop_event)
    except TrainingDependencyError as exc:
        print("Missing training dependencies: " + ", ".join(exc.missing), file=sys.stderr)
        return 2
    except ConfigError as exc:
        print(f"Invalid configuration: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 - the run is already recorded as failed
        print(f"Training failed: {exc}", file=sys.stderr)
        return 1
    if not args.json:
        elapsed = time.monotonic() - started
        print(f"Done in {elapsed / 60:.1f} min · output: {manifest.get('output')}")
    return 0


def _prepare(args: argparse.Namespace) -> int:
    try:
        report = analyze_dataset(args.dataset, max_seq_length=args.max_seq_length,
                                 validation_ratio=args.validation_ratio)
        prepared = prepare_dataset(args.dataset, args.output, max_seq_length=args.max_seq_length,
                                   validation_ratio=args.validation_ratio, dedupe=not args.no_dedupe,
                                   trim_long=args.trim_long)
    except Exception as exc:  # noqa: BLE001 - malformed datasets are user input
        print(f"Could not prepare the dataset: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(prepared.to_dict(), indent=2, default=str))
        return 0
    print(f"Rows: {report.rows} · valid: {report.valid} · invalid: {report.invalid} · "
          f"duplicates: {report.duplicates}")
    print(f"Tokens: avg {report.avg_tokens:.1f} · max {report.max_tokens} · "
          f"over limit: {report.very_long}")
    print(f"Split: {prepared.report.train} train / {prepared.report.validation} validation")
    print(f"Prepared: {prepared.train_path} and {prepared.validation_path}")
    return 0


def _plan(args: argparse.Namespace) -> int:
    try:
        config = TrainingConfig.from_file(args.config)
        config.validate()
        report = analyze_dataset(config.dataset.path, max_seq_length=config.dataset.max_seq_length,
                                 validation_ratio=config.dataset.validation_ratio)
    except Exception as exc:  # noqa: BLE001 - invalid recipes/datasets are user input
        print(f"Could not plan this run: {exc}", file=sys.stderr)
        return 2
    hardware = hardware_info()
    plan = estimate_plan(config, hardware, dataset_samples=report.valid,
                         dataset_avg_tokens=int(report.avg_tokens))
    if args.json:
        print(json.dumps({"plan": plan.to_dict(), "dataset": report.to_dict(),
                          "hardware": {"device": hardware.device, "gpu": hardware.gpu_name,
                                       "vram_total_bytes": hardware.vram_total_bytes,
                                       "ram_total_bytes": hardware.ram_total_bytes}}, indent=2, default=str))
        return 0
    _print_plan(plan)
    return 1 if plan.risk == "UNSUPPORTED" else 0


def _merge(args: argparse.Namespace) -> int:
    from noelle.training.trainer import merge_adapter

    try:
        output = merge_adapter(args.base, args.adapter, args.output,
                               trust_remote_code=args.trust_remote_code)
    except Exception as exc:  # noqa: BLE001 - CLI reports the failure and returns non-zero
        print(f"Merge failed: {exc}", file=sys.stderr)
        return 1
    print(output)
    return 0


def _list(args: argparse.Namespace) -> int:
    runs = TrainingStore().list(args.limit)
    if args.json:
        print(json.dumps(runs, indent=2, default=str))
        return 0
    if not runs:
        print("No training runs yet.")
        return 0
    for run in runs:
        step = f"{run.get('step') or 0}/{run.get('total_steps') or '?'}"
        loss = run.get("loss")
        loss_text = f" loss={loss:.4f}" if isinstance(loss, (int, float)) else ""
        print(f"{run['id']}  {run['status']:<10} {run['mode']:<8} step {step}{loss_text}")
    return 0


def _doctor(args: argparse.Namespace) -> int:
    import importlib

    from noelle.training.trainer import dependency_report

    report = dependency_report()
    load_errors: dict[str, str] = {}
    for name in ("torch", "transformers", "datasets", "accelerate", "peft", "trl"):
        try:
            importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001 - report the exact import problem
            load_errors[name] = str(exc)
    report["load_errors"] = load_errors
    compatible = report["compatible"] and not load_errors
    report["compatible"] = compatible
    if args.json:
        print(json.dumps(report, indent=2, default=str))
    elif compatible:
        versions = ", ".join(f"{name} {version}" for name, version in report["versions"].items() if version)
        print(f"Training packages are ready: {versions}")
        if not report["bitsandbytes"]:
            print("bitsandbytes is not installed; LoRA and full fine-tuning are available, QLoRA is not.")
    else:
        problems = list(report["missing"]) + [f"{name} ({error})" for name, error in load_errors.items()
                                             if name not in report["missing"]]
        print("Training packages are not usable: " + ", ".join(problems), file=sys.stderr)
        print("Create an environment with the training profile, or install them for this Python.",
              file=sys.stderr)
    return 0 if compatible else 1


def _status(args: argparse.Namespace) -> int:
    store = TrainingStore()
    run = store.get(args.run_id)
    if run is None:
        print(f"Training run not found: {args.run_id}", file=sys.stderr)
        return 1
    if args.json:
        payload = {**run, "checkpoints": store.checkpoints(args.run_id),
                   "metrics": store.metrics(args.run_id, limit=500)}
        print(json.dumps(payload, indent=2, default=str))
        return 0
    print(f"Run: {run['id']}")
    print(f"Status: {run['status']}  Mode: {run['mode']}  Model: {run.get('model_id')}")
    print(f"Step: {run.get('step') or 0}/{run.get('total_steps') or '?'}  Loss: {run.get('loss')}")
    if run.get("error"):
        print(f"Error: {run['error']}")
    if run.get("output_path"):
        print(f"Output: {run['output_path']}")
    for checkpoint in store.checkpoints(args.run_id):
        print(f"  checkpoint {checkpoint.get('step')}: {checkpoint['path']}")
    return 0


def main(args: argparse.Namespace) -> int:
    command = getattr(args, "train_command", None)
    if command == "run":
        return _run(args)
    if command == "prepare":
        return _prepare(args)
    if command == "plan":
        return _plan(args)
    if command == "merge":
        return _merge(args)
    if command == "doctor":
        return _doctor(args)
    if command == "list":
        return _list(args)
    if command == "status":
        return _status(args)
    print("Usage: noelle train {run,prepare,plan,merge,doctor,list,status}", file=sys.stderr)
    return 1