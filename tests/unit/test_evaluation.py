"""Evaluation Lab core: dataset parsing, local scorers, summaries, thresholds and experiment storage."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from noelle.runtime import evaluation as E


def case(**kw: object) -> E.EvalCase:
    return E.EvalCase.from_dict({"prompt": "q", **kw})


# --------------------------------------------------------------------- datasets
def test_builtin_catalog_loads_every_dataset() -> None:
    catalog = {item["name"]: item for item in E.builtin_catalog()}
    assert {"reasoning-math", "json-output", "instruction-following", "sentiment", "indonesian-qa",
            "translation-en-id", "coding-python", "chat-latency"} <= set(catalog)
    for name in catalog:
        dataset = E.load_dataset(f"builtin:{name}")
        assert dataset.cases and len(dataset.digest) == 64 and dataset.source == f"builtin:{name}"


def test_dataset_errors_point_at_the_line(tmp_path: Path) -> None:
    with pytest.raises(E.DatasetError, match="line 2: not valid JSON"):
        E.parse_dataset('{"prompt": "a"}\n{oops', source="x")
    with pytest.raises(E.DatasetError, match="line 1: case 1: needs a non-empty 'prompt'"):
        E.parse_dataset('{"expected": 1}', source="x")
    with pytest.raises(E.DatasetError, match="unknown scorer"):
        E.parse_dataset('{"prompt": "a", "scorer": "magic", "expected": 1}', source="x")
    with pytest.raises(E.DatasetError, match="duplicate case id 'a'"):
        E.parse_dataset('{"id": "a", "prompt": "x"}\n{"id": "a", "prompt": "y"}', source="x")
    with pytest.raises(E.DatasetError, match="invalid regex"):
        E.parse_dataset('{"prompt": "a", "scorer": "regex", "expected": "("}', source="x")
    with pytest.raises(E.DatasetError, match="no built-in dataset"):
        E.load_dataset("builtin:../../etc/passwd")
    path = tmp_path / "mine.jsonl"
    path.write_text('// comment\n\n{"prompt": "a", "expected": "b"}\n', encoding="utf-8")
    loaded = E.load_dataset(path)
    assert loaded.name == "mine" and loaded.cases[0].scorer == "contains" and loaded.cases[0].id == "case-1"


def test_digest_changes_with_content() -> None:
    a = E.parse_dataset('{"prompt": "a"}', source="x")
    b = E.parse_dataset('{"prompt": "b"}', source="x")
    assert a.digest != b.digest and a.digest == E.parse_dataset('{"prompt": "a"}', source="y").digest


# ----------------------------------------------------------------------- scorers
@pytest.mark.parametrize("output, ok", [("The answer is 21.", True), ("21", True), ("I think 1,000 then 21", True),
                                        ("twenty-one", False), ("<think>maybe 20</think>21", True)])
def test_numeric(output: str, ok: bool) -> None:
    assert E.score_case(case(expected=21, scorer="numeric"), output).passed is ok


def test_exact_contains_and_choice() -> None:
    assert E.score_case(case(expected="Paris", scorer="exact"), '  "paris."  ').passed
    assert not E.score_case(case(expected="Paris", scorer="exact"), "It is Paris").passed
    assert E.score_case(case(expected=["Soekarno", "Sukarno"], scorer="contains"), "Ir. Sukarno").passed
    partial = E.score_case(case(expected=["buku", "meja"], scorer="contains_all"), "Buku itu ada di atas lantai")
    assert partial.score == 0.5 and not partial.passed
    sentiment = case(expected="negative", scorer="choice", choices=["positive", "negative", "neutral"])
    assert E.score_case(sentiment, "Negative. Definitely not positive.").passed
    assert not E.score_case(sentiment, "positive, not negative").passed  # the first label counts
    assert not E.score_case(sentiment, "I cannot tell").passed


def test_regex_flags_and_forbidden_patterns() -> None:
    lower = case(expected=["\\bsea\\b", "^[^A-Z]+$"], scorer="regex", flags=["s"])
    assert E.score_case(lower, "the sea is calm today.").passed
    assert not E.score_case(lower, "The sea is calm today.").passed
    no_max = case(expected=["def\\s+f"], scorer="regex", forbid=["\\bmax\\s*\\("])
    assert E.score_case(no_max, "def f(xs):\n    return sorted(xs)[-1]").passed
    cheat = E.score_case(no_max, "def f(xs):\n    return max(xs)")
    assert not cheat.passed and cheat.score == 0.0 and "forbidden" in cheat.detail


def test_json_scorer_partial_credit() -> None:
    spec = {"type": "object", "required": ["name", "age"], "types": {"age": "integer"}, "equals": {"name": "Alice"}}
    good = E.score_case(case(expected=spec, scorer="json"), 'Sure!\n```json\n{"name": "alice", "age": 30}\n```')
    assert good.passed and good.score == 1.0
    wrong = E.score_case(case(expected=spec, scorer="json"), '{"name": "Bob", "age": "30"}')
    assert not wrong.passed and 0 < wrong.score < 1 and "'age' is integer" in wrong.detail
    assert E.score_case(case(expected=spec, scorer="json"), "no json here").score == 0.0
    assert E.score_case(case(expected={"type": "array", "value": [2, 3, 5]}, scorer="json"), "[2, 3, 5]").passed
    assert E.score_case(case(expected={"items": ["Saturday", "Sunday"]}, scorer="json"), '["sunday", "Saturday"]').passed
    assert not E.score_case(case(expected={"types": {"n": "integer"}}, scorer="json"), '{"n": true}').passed


def test_unscored_cases() -> None:
    result = E.score_case(case(), "anything")
    assert result.score is None and result.passed is None


def test_builtin_datasets_score_their_reference_answers() -> None:
    """Sanity: an ideal answer passes every scored built-in case (guards against broken expectations)."""
    ideal = {"m01": "21", "id01": "Soekarno", "t06": "Buku itu di atas meja", "s01": "positive",
             "j01": '{"name": "Alice", "age": 30}', "j05": "[2, 3, 5]", "i01": "YES", "i05": "the sea is calm.",
             "c06": "def max_of_list(xs):\n    best = xs[0]\n    for x in xs:\n        best = x if x > best else best\n"
                    "    return best"}
    cases = {c.id: c for name in ("reasoning-math", "indonesian-qa", "translation-en-id", "sentiment", "json-output",
                                  "instruction-following", "coding-python")
             for c in E.load_dataset(f"builtin:{name}").cases}
    for case_id, answer in ideal.items():
        assert E.score_case(cases[case_id], answer).passed, case_id


# ---------------------------------------------------------------------- summary
def test_summarize_and_thresholds() -> None:
    rows = [{"score": 1.0, "passed": True, "ttft_ms": 100, "tokens_per_second": 30, "latency_ms": 500,
             "completion_tokens": 10, "tags": ["math"]},
            {"score": 0.0, "passed": False, "ttft_ms": 300, "tokens_per_second": 20, "latency_ms": 700,
             "completion_tokens": 12, "tags": ["math"]},
            {"score": None, "passed": None, "ttft_ms": 200, "tokens_per_second": 25, "completion_tokens": 5},
            {"error": "boom"}]
    summary = E.summarize(rows)
    assert summary["accuracy"] == 0.5 and summary["scored"] == 2 and summary["errors"] == 1
    assert summary["ttft_ms_p50"] == 200 and summary["tokens_per_second"] == 25
    assert summary["by_tag"] == {"math": 0.5} and summary["completion_tokens"] == 27
    thresholds = [E.parse_threshold("accuracy>=0.5"), E.parse_threshold("tok_s >= 30"), E.parse_threshold("ttft<=150")]
    failures = E.check_thresholds(summary, thresholds)
    assert len(failures) == 2 and "tokens_per_second" in failures[0] and "ttft_ms_p50" in failures[1]
    with pytest.raises(E.DatasetError):
        E.parse_threshold("speed is fast")
    with pytest.raises(E.DatasetError, match="unknown metric"):
        E.parse_threshold("vibes>=3")


# ------------------------------------------------------------------- experiments
def test_old_registry_keeps_rows_and_gains_experiment_tables(tmp_path: Path) -> None:
    path = tmp_path / "registry.db"
    old = sqlite3.connect(path)
    old.executescript("""
        CREATE TABLE models (id INTEGER PRIMARY KEY AUTOINCREMENT, repo_id TEXT NOT NULL, revision TEXT,
            commit_hash TEXT, pipeline_tag TEXT, library_name TEXT, architecture TEXT, param_count INTEGER,
            license TEXT, local_path TEXT, precision TEXT NOT NULL DEFAULT 'auto', engine TEXT, gguf_file TEXT,
            source TEXT NOT NULL DEFAULT 'hub', base_model TEXT, size_bytes INTEGER,
            trust_remote_code INTEGER NOT NULL DEFAULT 0, downloaded_at REAL, last_used_at REAL, metadata TEXT,
            UNIQUE(repo_id, precision, source));
        CREATE TABLE evaluations (id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL, view INTEGER,
            model_id TEXT, prompt TEXT, system_prompt TEXT, params TEXT, output TEXT, status TEXT, ttft_ms REAL,
            tokens_per_second REAL, peak_memory_bytes INTEGER, precision TEXT, metrics TEXT, created_at REAL);
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
        INSERT INTO meta VALUES ('schema_version', '1');
        INSERT INTO models (repo_id, pipeline_tag) VALUES ('org/old', 'text-generation');
        INSERT INTO evaluations (run_id, model_id, status, tokens_per_second, ttft_ms, precision, metrics, created_at)
            VALUES ('r1', 'org/old', 'finished', 12.5, 300, 'fp16', '{}', 1);
    """)
    old.commit()
    old.close()
    from noelle.app.services.model_registry import ModelRegistry

    registry = ModelRegistry(path)
    assert registry.get_model("org/old") is not None
    assert registry.measurements("org/old")[0]["tokens_per_second"] == 12.5
    assert registry._rows("SELECT value FROM meta WHERE key='schema_version'")[0][0] == "2"
    assert registry.list_experiments() == []
    registry.close()


def test_record_export_and_matrix(tmp_path: Path) -> None:
    from noelle.app.services import experiments
    from noelle.app.services.model_registry import ModelRecord, ModelRegistry

    registry = ModelRegistry(tmp_path / "r.db")
    record = registry.upsert_model(ModelRecord(repo_id="org/m", pipeline_tag="text-generation", commit_hash="abc123",
                                               local_path=str(tmp_path), license="mit"))
    result = {"cases": 2, "total_ms": 900, "models": [{
        "view": 1, "model_id": "org/m", "precision": "auto", "status": "finished",
        "summary": {"accuracy": 0.5, "mean_score": 0.5, "ttft_ms_p50": 210.0, "tokens_per_second": 33.0, "errors": 0},
        "metrics": {"engine": "transformers", "device": "mps", "peak_bytes": 123456789, "load_ms": 800},
        "cases": [{"case_id": "a", "prompt": "q", "expected": 1, "output": "1", "score": 1.0, "passed": True},
                  {"case_id": "b", "prompt": "q", "expected": 2, "output": "3", "score": 0.0, "passed": False,
                   "detail": "answered 3, expected 2"}]}]}
    exp_id = experiments.record_evaluation(registry, result, dataset={"name": "math", "cases": 2, "digest": "d" * 64},
                                           params={"temperature": 0.0}, hardware={"device": "mps", "name": "M2"},
                                           models=[{"model_id": "org/m"}], records=[record])
    stored = registry.get_experiment(exp_id)
    assert stored["status"] == "finished" and stored["summary"]["best_model"] == "org/m"
    run = stored["runs"][0]
    assert run["provenance"]["commit_hash"] == "abc123" and run["cases"][1]["passed"] is False
    assert run["cases"][0]["expected"] == 1
    markdown = experiments.export_markdown(stored)
    assert "50.0%" in markdown and "abc123" in markdown and "answered 3, expected 2" in markdown
    assert json.loads(experiments.export_json(stored))["id"] == exp_id
    (row,) = registry.performance_matrix()
    assert row["tokens_per_second"] == 33.0 and row["accuracy"] == {"math": 0.5} and row["runs"] == 1
    assert registry.measurements("org/m")[0]["tokens_per_second"] == 33.0
    assert registry.delete_experiment(exp_id) and registry.get_experiment(exp_id) is None
    registry.close()
