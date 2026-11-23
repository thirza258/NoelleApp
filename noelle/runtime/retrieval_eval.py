"""Retrieval evaluation for knowledge bases: hit rate, recall, precision, MRR and nDCG at a cut-off k.

Pure Python (no Qt, no torch), so the desktop shell, scripts and tests can all use it.

A test case is a query plus the documents that answer it. Case files are JSON Lines, one object
per line::

    {"query": "How do I reset the router?", "relevant": ["manual.pdf", "faq/reset.md"]}

``relevant`` names documents, not chunks. A retrieved hit (a chunk, as ``RAG_QUERY`` returns it)
is relevant when the basename of its document equals an entry, or when an entry is a substring of
the document path. Both comparisons ignore case and treat ``/`` and ``\\`` alike, so one case file
works on every operating system.

Metrics use binary relevance on the top ``k`` hits of each case and are averaged over the cases
(macro average):

* **hit rate@k**: 1 when any of the top k hits is relevant, else 0.
* **recall@k**: the share of ``relevant`` entries matched by the top k hits. Each entry counts once,
  however many of its chunks were retrieved.
* **precision@k**: relevant hits among the top k, divided by k. Every relevant chunk counts.
* **MRR**: 1 / rank of the first relevant hit, or 0 when no top k hit is relevant.
* **nDCG@k**: DCG / ideal DCG with the discount ``log2(rank + 1)``. A hit gains 1 only when it
  matches an entry that no earlier hit matched, so further chunks of an already credited document
  gain 0. The ideal ranking puts ``min(len(relevant), k)`` relevant documents first, which keeps
  nDCG between 0 and 1.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

Hit = Mapping[str, Any] | str  # a RAG_QUERY hit ({"path", "source", ...}) or a bare document path

# (report attribute, label template) in display order
METRICS: tuple[tuple[str, str], ...] = (
    ("hit_rate", "Hit rate@{k}"),
    ("recall", "Recall@{k}"),
    ("precision", "Precision@{k}"),
    ("mrr", "MRR"),
    ("ndcg", "nDCG@{k}"),
)


def metric_label(key: str, k: int) -> str:
    return dict(METRICS)[key].format(k=k)


# ============================================================================
# cases and results
# ============================================================================


@dataclass(frozen=True)
class EvalCase:
    query: str
    relevant: tuple[str, ...]

    @classmethod
    def create(cls, query: Any, relevant: Any) -> EvalCase:
        """A validated case. Blank and duplicate ``relevant`` entries are dropped; at least one must remain."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("'query' must be a non-empty string")
        if isinstance(relevant, str) or not isinstance(relevant, (list, tuple)):
            raise ValueError("'relevant' must be a list of document names or paths")
        entries: list[str] = []
        seen: set[str] = set()
        for position, entry in enumerate(relevant, start=1):
            if not isinstance(entry, str):
                raise ValueError(f"'relevant' item {position} must be a string")
            key = _normalize(entry)
            if key and key not in seen:
                seen.add(key)
                entries.append(entry.strip())
        if not entries:
            raise ValueError("'relevant' needs at least one document name or path")
        return cls(query.strip(), tuple(entries))

    def to_dict(self) -> dict[str, Any]:
        return {"query": self.query, "relevant": list(self.relevant)}


@dataclass(frozen=True)
class CaseResult:
    case: EvalCase
    retrieved: tuple[str, ...]  # document name of each scored hit, best first
    relevant_ranks: tuple[int, ...]  # 1-based ranks of the relevant hits
    found: tuple[str, ...]  # ``relevant`` entries matched by the scored hits, in the order first matched
    hit: float
    recall: float
    precision: float
    reciprocal_rank: float
    ndcg: float

    @property
    def first_rank(self) -> int | None:
        return self.relevant_ranks[0] if self.relevant_ranks else None

    @property
    def missed(self) -> tuple[str, ...]:
        return tuple(entry for entry in self.case.relevant if entry not in self.found)


@dataclass(frozen=True)
class EvalReport:
    k: int
    cases: tuple[CaseResult, ...]
    hit_rate: float
    recall: float
    precision: float
    mrr: float
    ndcg: float

    @property
    def hits(self) -> int:
        """Number of cases with at least one relevant hit in the top k."""
        return sum(1 for result in self.cases if result.hit)

    def metric(self, key: str) -> float:
        """Mean metric by its :data:`METRICS` key."""
        return float(getattr(self, key))


# ============================================================================
# metrics
# ============================================================================


def matched_entries(hit: Hit, relevant: Sequence[str]) -> list[str]:
    """The ``relevant`` entries that ``hit`` matches: equal to its basename, or a substring of its path."""
    name, path = _document(hit)
    matches: list[str] = []
    for entry in relevant:
        key = _normalize(entry)
        if key and (key == name or key in path):
            matches.append(entry)
    return matches


def evaluate_case(case: EvalCase, hits: Sequence[Hit], k: int) -> CaseResult:
    """Score the ranked ``hits`` of one case; only the first ``k`` count."""
    k = _check_k(k)
    if not case.relevant:
        raise ValueError("a case needs at least one relevant document")
    top = list(hits)[:k]
    credited: list[str] = []
    ranks: list[int] = []
    dcg = 0.0
    for rank, hit in enumerate(top, start=1):
        matches = matched_entries(hit, case.relevant)
        if not matches:
            continue
        ranks.append(rank)
        fresh = [entry for entry in matches if entry not in credited]
        if fresh:
            credited.extend(fresh)
            dcg += 1.0 / math.log2(rank + 1)
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(case.relevant), k) + 1))
    return CaseResult(
        case=case,
        retrieved=tuple(_display_name(hit) for hit in top),
        relevant_ranks=tuple(ranks),
        found=tuple(credited),
        hit=1.0 if ranks else 0.0,
        recall=len(credited) / len(case.relevant),
        precision=len(ranks) / k,
        reciprocal_rank=1.0 / ranks[0] if ranks else 0.0,
        ndcg=dcg / ideal,
    )


def evaluate(cases: Sequence[EvalCase], hits_per_case: Sequence[Sequence[Hit]], k: int) -> EvalReport:
    """Per-case and mean metrics; ``hits_per_case[i]`` are the ranked hits retrieved for ``cases[i]``."""
    k = _check_k(k)
    if len(cases) != len(hits_per_case):
        raise ValueError(f"got hits for {len(hits_per_case)} queries but there are {len(cases)} cases")
    results = tuple(evaluate_case(case, hits, k) for case, hits in zip(cases, hits_per_case))

    def mean(attribute: str) -> float:
        return sum(getattr(result, attribute) for result in results) / len(results) if results else 0.0

    return EvalReport(k=k, cases=results, hit_rate=mean("hit"), recall=mean("recall"), precision=mean("precision"),
                      mrr=mean("reciprocal_rank"), ndcg=mean("ndcg"))


def _check_k(k: int) -> int:
    if int(k) < 1:
        raise ValueError("k must be at least 1")
    return int(k)


def _normalize(text: str) -> str:
    return text.strip().replace("\\", "/").casefold()


def _document(hit: Hit) -> tuple[str, str]:
    """Normalised ``(basename, path)`` of a hit's document."""
    if isinstance(hit, str):
        path, name = hit, ""
    else:
        path, name = str(hit.get("path") or ""), str(hit.get("source") or "")
    path = _normalize(path or name)
    return _normalize(name) or path.rsplit("/", 1)[-1], path


def _display_name(hit: Hit) -> str:
    if isinstance(hit, str):
        return hit.replace("\\", "/").rsplit("/", 1)[-1]
    return str(hit.get("source") or str(hit.get("path") or "").replace("\\", "/").rsplit("/", 1)[-1])


# ============================================================================
# case files (JSON Lines)
# ============================================================================


class CaseFileError(ValueError):
    """A malformed case file. ``line`` is the 1-based physical line number (blank lines count)."""

    def __init__(self, line: int, reason: str) -> None:
        super().__init__(f"Line {line}: {reason}")
        self.line = line
        self.reason = reason


def parse_cases(text: str) -> list[EvalCase]:
    """Cases from JSON Lines text: one ``{"query", "relevant"}`` object per line, blank lines skipped."""
    cases: list[EvalCase] = []
    lines = text.removeprefix("﻿").replace("\r\n", "\n").replace("\r", "\n").split("\n")
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CaseFileError(number, f"invalid JSON ({exc.msg} at column {exc.colno})") from None
        if not isinstance(data, dict):
            raise CaseFileError(number, 'expected an object like {"query": "...", "relevant": ["file.pdf"]}')
        try:
            cases.append(EvalCase.create(data.get("query"), data.get("relevant")))
        except ValueError as exc:
            raise CaseFileError(number, str(exc)) from None
    return cases


def dumps_cases(cases: Iterable[EvalCase]) -> str:
    return "".join(json.dumps(case.to_dict(), ensure_ascii=False) + "\n" for case in cases)


def load_cases(path: str | Path) -> list[EvalCase]:
    data = Path(path).read_bytes()
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise CaseFileError(data[:exc.start].count(b"\n") + 1, "the file is not UTF-8 text") from None
    return parse_cases(text)


def save_cases(path: str | Path, cases: Iterable[EvalCase]) -> None:
    Path(path).write_text(dumps_cases(cases), encoding="utf-8", newline="\n")
