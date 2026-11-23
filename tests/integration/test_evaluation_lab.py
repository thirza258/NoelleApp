"""Evaluation Lab end to end: RUN_EVAL on a real worker, scored and stored as an experiment."""

from __future__ import annotations

from typing import Any

import pytest

from noelle.runtime.hub_cache import cached_snapshot_path
from noelle.runtime.protocol import Command, Event

pytestmark = pytest.mark.integration

SMALL = "HuggingFaceTB/SmolLM2-135M-Instruct"


def test_run_eval_scores_cases_and_records_experiment(worker_process: Any) -> None:
    if cached_snapshot_path(SMALL) is None:
        pytest.skip(f"{SMALL} is not in the local Hugging Face cache")
    from noelle.app.services import experiments
    from noelle.app.services.model_registry import ModelRegistry
    from noelle.runtime import evaluation

    dataset = evaluation.parse_dataset("\n".join([
        '{"_dataset": {"name": "tiny", "title": "Tiny"}}',
        '{"id": "a", "prompt": "Reply with the word yes.", "expected": "yes", "scorer": "contains", "max_new_tokens": 8}',
        '{"id": "b", "prompt": "What is 2 + 2? Answer with just the number.", "expected": 4, "scorer": "numeric", '
        '"max_new_tokens": 8}',
        '{"id": "c", "prompt": "Say hello.", "scorer": "none", "max_new_tokens": 8}',
    ]), source="test")
    payload = {"models": [{"model_id": SMALL, "precision": "auto"}],
               "cases": [case.to_dict() for case in dataset.cases], "params": {"max_new_tokens": 16}}
    events: list[dict[str, Any]] = []
    for message in worker_process.client().stream(Command.RUN_EVAL, payload, timeout=600):
        events.append(message)
    kinds = [m["event"] for m in events]
    assert Event.QUEUE_PLAN in kinds and Event.VIEW_STARTED in kinds and Event.VIEW_COMPLETE in kinds
    progress = [m["payload"] for m in events if m["event"] == Event.PROGRESS and m["payload"].get("case")]
    assert [p["case"]["case_id"] for p in progress] == ["a", "b", "c"] and progress[-1]["current"] == 3
    final = events[-1]
    assert final["event"] == Event.GENERATION_COMPLETE, final
    (model,) = final["payload"]["models"]
    assert model["status"] == "finished" and len(model["cases"]) == 3
    summary = model["summary"]
    assert summary["cases"] == 3 and summary["scored"] == 2 and summary["tokens_per_second"] > 0
    assert model["metrics"]["engine"] == "transformers" and model["metrics"]["peak_bytes"] > 0
    assert model["cases"][2]["score"] is None  # performance-only case

    registry = ModelRegistry(":memory:")
    experiment_id = experiments.record_evaluation(
        registry, final["payload"], dataset=dataset.meta(), params=payload["params"],
        hardware={"device": "test"}, models=payload["models"], records=[None])
    stored = registry.get_experiment(experiment_id)
    assert stored["status"] == "finished" and len(stored["runs"][0]["cases"]) == 3
    assert stored["dataset"]["digest"] == dataset.digest
    assert "| Model |" in experiments.export_markdown(stored)


def test_run_eval_rejects_bad_requests(worker_process: Any) -> None:
    from noelle.runtime.client import WorkerError

    client = worker_process.client()
    with pytest.raises(WorkerError, match="at least one model"):
        client.call(Command.RUN_EVAL, {"models": [], "cases": [{"prompt": "x"}]})
    with pytest.raises(WorkerError, match="Invalid dataset"):
        client.call(Command.RUN_EVAL, {"models": [{"model_id": SMALL}], "cases": [{"prompt": "x", "scorer": "nope",
                                                                                   "expected": 1}]})
