"""Retrieval evaluation: relevance matching, metrics against hand-computed values, and JSONL case files."""

from __future__ import annotations

import json
from math import log2
from pathlib import Path

import pytest

from noelle.runtime import retrieval_eval as rev
from noelle.runtime.retrieval_eval import CaseFileError, EvalCase


def _hit(path: str, source: str | None = None) -> dict[str, object]:
    return {"path": path, "source": source if source is not None else path.replace("\\", "/").rsplit("/", 1)[-1],
            "chunk_id": 1, "page": None, "text": "...", "score": 0.5}


# ------------------------------------------------------------------- relevance


def test_relevance_uses_basename_or_path_substring_ignoring_case_and_separators() -> None:
    relevant = ["Guide.PDF", "legal/terms", "faq\\reset.md"]
    assert rev.matched_entries(_hit("/docs/guide.pdf"), relevant) == ["Guide.PDF"]
    assert rev.matched_entries(_hit("C:\\Docs\\Legal\\Terms-of-sale.docx"), relevant) == ["legal/terms"]
    assert rev.matched_entries(_hit("/kb/FAQ/Reset.md"), relevant) == ["faq\\reset.md"]
    assert rev.matched_entries(_hit("/docs/guide.pdf.bak"), ["guide.pdf"]) == ["guide.pdf"]  # substring of the path
    assert rev.matched_entries(_hit("/docs/notes.txt"), relevant) == []
    # the basename comes from the hit's source (the engine computed it on the ingesting OS)
    assert rev.matched_entries({"path": "", "source": "Report.md"}, ["report.md"]) == ["report.md"]
    assert rev.matched_entries("/plain/path/a.md", ["A.MD"]) == ["A.MD"]
    assert rev.matched_entries({}, ["a.md"]) == []


# --------------------------------------------------------------------- metrics


def test_case_metrics_match_hand_computed_values() -> None:
    case = EvalCase.create("q", ["a.md", "b.pdf", "c.txt"])
    hits = [_hit("/x/other.md"), _hit("/d/a.md"), _hit("/d/a.md"), _hit("/x/more.md"), _hit("/d/b.pdf")]
    result = rev.evaluate_case(case, hits, k=5)
    assert result.hit == 1.0
    assert result.recall == pytest.approx(2 / 3)  # a.md counts once although two of its chunks were retrieved
    assert result.precision == pytest.approx(3 / 5)  # every relevant chunk counts
    assert result.reciprocal_rank == pytest.approx(1 / 2)
    dcg = 1 / log2(3) + 1 / log2(6)  # ranks 2 and 5 credit new documents; rank 3 repeats a.md
    ideal = 1 / log2(2) + 1 / log2(3) + 1 / log2(4)
    assert result.ndcg == pytest.approx(dcg / ideal)
    assert result.relevant_ranks == (2, 3, 5) and result.first_rank == 2
    assert result.found == ("a.md", "b.pdf") and result.missed == ("c.txt",)
    assert result.retrieved == ("other.md", "a.md", "a.md", "more.md", "b.pdf")


def test_ndcg_stays_at_most_one_when_one_document_fills_the_ranking() -> None:
    result = rev.evaluate_case(EvalCase.create("q", ["only.md"]), [_hit("/only.md")] * 3, k=3)
    assert result.ndcg == pytest.approx(1.0) and result.precision == pytest.approx(1.0) and result.recall == 1.0


def test_only_the_top_k_hits_count() -> None:
    case = EvalCase.create("q", ["z.md"])
    hits = [_hit("/a.md"), _hit("/b.md"), _hit("/z.md")]
    cut = rev.evaluate_case(case, hits, k=2)
    assert (cut.hit, cut.recall, cut.precision, cut.reciprocal_rank, cut.ndcg) == (0.0, 0.0, 0.0, 0.0, 0.0)
    assert cut.retrieved == ("a.md", "b.md") and cut.first_rank is None
    full = rev.evaluate_case(case, hits, k=3)
    assert full.reciprocal_rank == pytest.approx(1 / 3) and full.ndcg == pytest.approx(1 / log2(4))


def test_precision_divides_by_k_even_with_fewer_hits() -> None:
    result = rev.evaluate_case(EvalCase.create("q", ["a.md", "b.md"]), [_hit("/a.md")], k=4)
    assert result.precision == pytest.approx(1 / 4) and result.recall == pytest.approx(1 / 2)
    assert result.ndcg == pytest.approx(1 / (1 + 1 / log2(3)))


def test_report_averages_over_cases() -> None:
    cases = [EvalCase.create("first", ["a.md"]), EvalCase.create("second", ["b.md", "c.md"])]
    hits = [[_hit("/a.md"), _hit("/x.md")], [_hit("/x.md"), _hit("/c.md")]]
    report = rev.evaluate(cases, hits, k=2)
    assert report.k == 2 and len(report.cases) == 2 and report.hits == 2
    assert report.hit_rate == pytest.approx(1.0)
    assert report.recall == pytest.approx((1 + 1 / 2) / 2)
    assert report.precision == pytest.approx((1 / 2 + 1 / 2) / 2)
    assert report.mrr == pytest.approx((1 + 1 / 2) / 2)
    assert report.ndcg == pytest.approx((1 + (1 / log2(3)) / (1 + 1 / log2(3))) / 2)
    assert report.metric("mrr") == report.mrr
    assert [rev.metric_label(key, 5) for key, _ in rev.METRICS] == [
        "Hit rate@5", "Recall@5", "Precision@5", "MRR", "nDCG@5"]
    empty = rev.evaluate([], [], k=3)
    assert (empty.hit_rate, empty.recall, empty.precision, empty.mrr, empty.ndcg, empty.hits) == (0, 0, 0, 0, 0, 0)


def test_evaluate_rejects_bad_arguments() -> None:
    case = EvalCase.create("q", ["a.md"])
    with pytest.raises(ValueError, match="k must be at least 1"):
        rev.evaluate([case], [[]], k=0)
    with pytest.raises(ValueError, match="hits for 2 queries but there are 1 cases"):
        rev.evaluate([case], [[], []], k=3)
    with pytest.raises(ValueError, match="at least one relevant"):
        rev.evaluate_case(EvalCase("q", ()), [], k=3)


def test_cases_are_validated_and_normalised() -> None:
    case = EvalCase.create("  How?  ", [" a.md ", "A.MD", "", "docs/b.md", "docs\\B.md"])
    assert case == EvalCase("How?", ("a.md", "docs/b.md"))  # stripped; blanks and case/separator duplicates dropped
    assert case.to_dict() == {"query": "How?", "relevant": ["a.md", "docs/b.md"]}
    for query, relevant, message in (("", ["a.md"], "'query' must be a non-empty string"),
                                     (7, ["a.md"], "'query' must be a non-empty string"),
                                     ("q", "a.md", "'relevant' must be a list"),
                                     ("q", None, "'relevant' must be a list"),
                                     ("q", ["a.md", 3], "'relevant' item 2 must be a string"),
                                     ("q", [" ", ""], "'relevant' needs at least one")):
        with pytest.raises(ValueError, match=message):
            EvalCase.create(query, relevant)


# ------------------------------------------------------------------ case files


def test_jsonl_round_trip(tmp_path: Path) -> None:
    cases = [EvalCase.create("Où est la notice ?", ["notice.pdf"]),
             EvalCase.create("line\u2028separator inside", ["a.md", "dir/b.md"])]
    path = tmp_path / "cases.jsonl"
    rev.save_cases(path, cases)
    raw = path.read_bytes()
    assert raw.count(b"\n") == 2 and b"\r" not in raw and "Où".encode() in raw  # UTF-8, not \u escapes
    assert rev.load_cases(path) == cases
    assert rev.parse_cases(rev.dumps_cases(cases)) == cases


def test_parser_skips_blank_lines_and_accepts_bom_and_crlf(tmp_path: Path) -> None:
    text = "\ufeff" + json.dumps({"query": "a", "relevant": ["x.md"], "note": "extra keys are ignored"}) + "\r\n"
    text += "   \r\n\r\n" + json.dumps({"query": "b", "relevant": ["y.md"]}) + "\r\n"
    assert [case.query for case in rev.parse_cases(text)] == ["a", "b"]
    path = tmp_path / "bom.jsonl"
    path.write_bytes(text.encode("utf-8"))
    assert len(rev.load_cases(path)) == 2
    assert rev.parse_cases("") == [] and rev.parse_cases("\n \n") == []


@pytest.mark.parametrize(("text", "line", "reason"), [
    ('{"query": "a", "relevant": ["x"]}\n\n{"query": "b", "relevant": ["y"]\n', 3, "invalid JSON"),
    ('["not", "an", "object"]\n', 1, "expected an object"),
    ('{"query": "a", "relevant": ["x"]}\n{"relevant": ["y"]}\n', 2, "'query' must be a non-empty string"),
    ('\n{"query": "a", "relevant": "x.md"}\n', 2, "'relevant' must be a list"),
    ('{"query": "a", "relevant": []}\n', 1, "'relevant' needs at least one"),
    ('{"query": "a", "relevant": ["x", null]}\n', 1, "'relevant' item 2 must be a string"),
])
def test_parser_errors_carry_line_numbers(text: str, line: int, reason: str) -> None:
    with pytest.raises(CaseFileError) as info:
        rev.parse_cases(text)
    assert info.value.line == line and reason in info.value.reason
    assert str(info.value).startswith(f"Line {line}: ")


def test_loader_reports_undecodable_lines(tmp_path: Path) -> None:
    path = tmp_path / "latin1.jsonl"
    path.write_bytes(b'{"query": "a", "relevant": ["x"]}\n{"query": "caf\xe9", "relevant": ["y"]}\n')
    with pytest.raises(CaseFileError, match="Line 2: the file is not UTF-8 text"):
        rev.load_cases(path)
