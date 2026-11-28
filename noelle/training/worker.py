"""Training worker entry point (``noelle train-worker``).

The desktop shell launches one of these per run. Stdout carries newline-delimited
JSON events only; library chatter is redirected to stderr and the log file.
"""

from __future__ import annotations

import json
import logging
import signal
import sys
import threading
import traceback
from pathlib import Path
from typing import Any

from noelle.training.config import ConfigError, TrainingConfig
from noelle.training.store import TrainingStore
from noelle.training.trainer import TrainingDependencyError, run_training

log = logging.getLogger("noelle.training.worker")


def add_arguments(parser: Any) -> None:
    parser.add_argument("--run-id", required=True, help="training run id created by the desktop app")
    parser.add_argument("--config", required=True, help="path to the run's config.json")


def main(args: Any) -> int:
    logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # Libraries (transformers, TRL) print to stdout; only our JSON lines may go there.
    events = sys.stdout
    sys.stdout = sys.stderr

    def emit(event: dict[str, Any]) -> None:
        try:
            events.write(json.dumps(event, default=str) + "\n")
            events.flush()
        except (OSError, ValueError):
            pass

    stop_event = threading.Event()

    def request_stop(_signum: int, _frame: Any) -> None:
        stop_event.set()
        emit({"type": "training.warning", "message": "Stop requested; finishing the current step."})

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    run_id = str(args.run_id)
    store = TrainingStore()
    try:
        config = TrainingConfig.from_file(Path(args.config))
        config.validate()
    except Exception as exc:  # noqa: BLE001 - a hand-edited recipe must not crash the worker
        message = f"Invalid training configuration: {exc}"
        store.update(run_id, status="failed", error=message)
        emit({"type": "training.error", "code": "CONFIG", "message": message})
        return 2
    try:
        run_training(config, run_id, store, emit, stop_event)
    except TrainingDependencyError:
        return 2
    except ConfigError as exc:
        store.update(run_id, status="failed", error=str(exc))
        emit({"type": "training.error", "code": "CONFIG", "message": str(exc)})
        return 2
    except Exception as exc:  # noqa: BLE001 - report every failure to the desktop shell
        detail = traceback.format_exc()
        log.error("training run %s failed: %s\n%s", run_id, exc, detail)
        run = store.get(run_id) or {}
        if run.get("status") not in ("failed", "oom", "stopped", "completed"):
            store.update(run_id, status="failed", error=str(exc))
            emit({"type": "training.error", "code": "TRAINING_ERROR", "message": f"Training failed: {exc}",
                  "detail": detail})
        return 1
    return 0