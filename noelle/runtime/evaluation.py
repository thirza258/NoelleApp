"""Evaluation Lab core: datasets, deterministic local scorers and result summaries.

A dataset is a JSONL file. An optional first line describes it::

    {"_dataset": {"name": "reasoning-math", "title": "Arithmetic word problems", "category": "reasoning",
                  "description": "..."}}

Every other line is one case::

    {"id": "m1", "prompt": "What is 17 x 23? Answer with just the number.", "expected": 391, "scorer": "numeric"}

Fields: ``prompt`` or ``messages`` (OpenAI-style), optional ``system``, ``expected``,
``scorer`` (exact | contains | contains_all | regex | numeric | choice | json | none),
``choices`` (for ``choice``), ``forbid`` (regexes that must not match, for ``regex``),
``flags`` (regex flags, default ``["i", "m"]``), ``tolerance`` (``numeric``),
``tags`` and ``max_new_tokens``.

Scoring is local and deterministic — no model judges another — so a run is
reproducible from the dataset digest, the model revision and the generation
parameters. Pure Python (no torch, no Qt): the worker scores cases as they
finish and the UI/CLI reuse the same summaries.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

BUILTIN_DIR = Path(__file__).resolve().parents[1] / "config" / "eval_datasets"
SCORERS = ("exact", "contains", "contains_all", "regex", "numeric", "choice", "json", "none")
MAX_CASES = 5000
MAX_OUTPUT_CHARS = 8000
_THINK = re.compile(r"<think>.*?(</think>|$)", re.DOTALL | re.IGNORECASE)
_FENCE = re.compile(r"^\s*```[a-zA-Z0-9_-]*\s*\n?(.*?)\n?\s*```\s*$", re.DOTALL)
_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
_FLAG_MAP = {"i": re.IGNORECASE, "m": re.MULTILINE, "s": re.DOTALL, "x": re.VERBOSE}


class DatasetError(ValueError):
    pass


@dataclass
class EvalCase:
    id: str
    prompt: str | None = None
    messages: list[dict[str, Any]] | None = None
    system: str | None = None
    expected: Any = None
    scorer: str = "none"
    choices: list[str] = field(default_factory=list)
    forbid: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=lambda: ["i", "m"])
    tolerance: float | None = None
    tags: list[str] = field(default_factory=list)
    max_new_tokens: int | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any], index: int = 0) -> EvalCase:
        if not isinstance(data, dict):
            raise DatasetError(f"case {index + 1}: expected an object")
        prompt, messages = data.get("prompt"), data.get("messages")
        if not (isinstance(prompt, str) and prompt.strip()) and not (isinstance(messages, list) and messages):
            raise DatasetError(f"case {index + 1}: needs a non-empty 'prompt' or 'messages'")
        scorer = str(data.get("scorer") or ("none" if data.get("expected") is None else "contains"))
        if scorer not in SCORERS:
            raise DatasetError(f"case {index + 1}: unknown scorer '{scorer}' (use one of {', '.join(SCORERS)})")
        if scorer not in ("none",) and data.get("expected") is None:
            raise DatasetError(f"case {index + 1}: scorer '{scorer}' needs 'expected'")
        if scorer == "choice" and not data.get("choices"):
            raise DatasetError(f"case {index + 1}: scorer 'choice' needs 'choices'")
        if scorer == "regex":
            for pattern in _as_list(data.get("expected")) + _as_list(data.get("forbid")):
                try:
                    re.compile(str(pattern))
                except re.error as exc:
                    raise DatasetError(f"case {index + 1}: invalid regex {pattern!r}: {exc}") from exc
        max_new = data.get("max_new_tokens")
        return cls(
            id=str(data.get("id") or f"case-{index + 1}"),
            prompt=prompt if isinstance(prompt, str) else None,
            messages=messages if isinstance(messages, list) else None,
            system=data.get("system") if isinstance(data.get("system"), str) else None,
            expected=data.get("expected"),
            scorer=scorer,
            choices=[str(c) for c in data.get("choices") or []],
            forbid=[str(f) for f in _as_list(data.get("forbid"))],
            flags=[str(f) for f in data.get("flags", ["i", "m"])],
            tolerance=float(data["tolerance"]) if data.get("tolerance") is not None else None,
            tags=[str(t) for t in data.get("tags") or []],
            max_new_tokens=int(max_new) if max_new is not None else None,
        )

    def chat_messages(self) -> list[dict[str, Any]]:
        if self.messages:
            return [dict(m) for m in self.messages]
        return [{"role": "user", "content": self.prompt or ""}]

    def display_prompt(self) -> str:
        if self.prompt:
            return self.prompt
        for message in reversed(self.messages or []):
            if message.get("role") == "user":
                return str(message.get("content") or "")
        return ""

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v not in (None, [], "") or k in ("id", "scorer")}


@dataclass
class EvalDataset:
    name: str
    title: str
    description: str
    category: str
    cases: list[EvalCase]
    source: str  # "builtin:<name>" or a file path
    digest: str

    def meta(self) -> dict[str, Any]:
        return {"name": self.name, "title": self.title, "description": self.description, "category": self.category,
                "source": self.source, "digest": self.digest, "cases": len(self.cases)}


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------


def parse_dataset(text: str, *, source: str, default_name: str = "dataset") -> EvalDataset:
    meta: dict[str, Any] = {}
    cases: list[EvalCase] = []
    for number, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        try:
            obj = json.loads(line)
        except ValueError as exc:
            raise DatasetError(f"line {number}: not valid JSON ({exc.msg})") from exc
        if isinstance(obj, dict) and "_dataset" in obj:
            if not isinstance(obj["_dataset"], dict):
                raise DatasetError(f"line {number}: '_dataset' must be an object")
            meta = obj["_dataset"]
            continue
        try:
            cases.append(EvalCase.from_dict(obj, len(cases)))
        except DatasetError as exc:
            raise DatasetError(f"line {number}: {exc}") from exc
        if len(cases) > MAX_CASES:
            raise DatasetError(f"more than {MAX_CASES} cases; split the dataset")
    if not cases:
        raise DatasetError("the dataset has no cases")
    ids = [c.id for c in cases]
    if len(set(ids)) != len(ids):
        duplicate = next(i for i in ids if ids.count(i) > 1)
        raise DatasetError(f"duplicate case id '{duplicate}'")
    digest = hashlib.sha256(json.dumps([c.to_dict() for c in cases], sort_keys=True).encode()).hexdigest()
    name = str(meta.get("name") or default_name)
    return EvalDataset(name=name, title=str(meta.get("title") or name), description=str(meta.get("description") or ""),
                       category=str(meta.get("category") or "custom"), cases=cases, source=source, digest=digest)


def load_dataset(ref: str | Path) -> EvalDataset:
    """``builtin:<name>`` or a path to a ``.jsonl`` file."""
    text_ref = str(ref)
    if text_ref.startswith("builtin:"):
        name = text_ref.split(":", 1)[1]
        path = BUILTIN_DIR / f"{name}.jsonl"
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", name) or not path.is_file():
            available = ", ".join(d["name"] for d in builtin_catalog())
            raise DatasetError(f"no built-in dataset '{name}' (available: {available})")
        return parse_dataset(path.read_text(encoding="utf-8"), source=text_ref, default_name=name)
    path = Path(text_ref).expanduser()
    if not path.is_file():
        raise DatasetError(f"dataset file not found: {path}")
    if path.stat().st_size > 50 * 1024 * 1024:
        raise DatasetError("dataset file is larger than 50 MB")
    return parse_dataset(path.read_text(encoding="utf-8"), source=str(path), default_name=path.stem)


def builtin_catalog() -> list[dict[str, Any]]:
    """Metadata of the datasets shipped with Noelle (cases are loaded on demand)."""
    items = []
    for path in sorted(BUILTIN_DIR.glob("*.jsonl")):
        try:
            dataset = parse_dataset(path.read_text(encoding="utf-8"), source=f"builtin:{path.stem}",
                                    default_name=path.stem)
        except (OSError, DatasetError):
            continue
        items.append(dataset.meta())
    return items


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------


@dataclass
class CaseScore:
    score: float | None  # 0..1, None when the case is not scored
    passed: bool | None
    detail: str


def clean_output(text: str) -> str:
    """Drop reasoning blocks (``<think>…</think>``) that some models emit before the answer."""
    return _THINK.sub("", text or "").strip()


def score_case(case: EvalCase, output: str) -> CaseScore:
    text = clean_output(output)
    scorer = case.scorer
    if scorer == "none":
        return CaseScore(None, None, "not scored (performance only)")
    try:
        if scorer == "exact":
            ok = any(_normalize(text) == _normalize(str(e)) for e in _as_list(case.expected))
            return _binary(ok, "exact match" if ok else f"expected {_show(case.expected)}")
        if scorer in ("contains", "contains_all"):
            wanted = [str(e) for e in _as_list(case.expected)]
            haystack = _squash(text)
            found = [w for w in wanted if _squash(w) in haystack]
            ok = len(found) == len(wanted) if scorer == "contains_all" else bool(found)
            if scorer == "contains_all" and wanted:
                return CaseScore(len(found) / len(wanted), ok, f"found {len(found)}/{len(wanted)}: {_show(found)}")
            return _binary(ok, f"found {_show(found[0])}" if found else f"none of {_show(wanted)}")
        if scorer == "regex":
            flags = 0
            for flag in case.flags:
                flags |= _FLAG_MAP.get(flag, 0)
            patterns = [str(p) for p in _as_list(case.expected)]
            matched = [p for p in patterns if re.search(p, text, flags)]
            forbidden = [p for p in case.forbid if re.search(p, text, flags)]
            ok = len(matched) == len(patterns) and not forbidden
            detail = f"matched {len(matched)}/{len(patterns)} patterns" + (f"; forbidden {forbidden}" if forbidden else "")
            score = (len(matched) / len(patterns) if patterns else 1.0) * (0.0 if forbidden else 1.0)
            return CaseScore(score, ok, detail)
        if scorer == "numeric":
            value = last_number(text)
            if value is None:
                return _binary(False, "no number in the answer")
            expected = float(case.expected)
            tolerance = case.tolerance if case.tolerance is not None else max(1e-6, abs(expected) * 1e-6)
            ok = math.isclose(value, expected, abs_tol=tolerance)
            return _binary(ok, f"answered {value:g}, expected {expected:g}")
        if scorer == "choice":
            picked = first_choice(text, case.choices)
            ok = picked is not None and picked.lower() == str(case.expected).lower()
            return _binary(ok, f"picked {picked!r}, expected {case.expected!r}")
        if scorer == "json":
            return _score_json(text, case.expected)
    except (TypeError, ValueError, re.error) as exc:
        return CaseScore(0.0, False, f"scoring error: {exc}")
    return CaseScore(None, None, f"unknown scorer {scorer}")


def last_number(text: str) -> float | None:
    numbers = _NUMBER.findall(text)
    for raw in reversed(numbers):
        try:
            return float(raw.replace(",", ""))
        except ValueError:
            continue
    return None


def first_choice(text: str, choices: Iterable[str]) -> str | None:
    best: tuple[int, str] | None = None
    for choice in choices:
        match = re.search(rf"(?<![\w-]){re.escape(choice)}(?![\w-])", text, re.IGNORECASE)
        if match and (best is None or match.start() < best[0]):
            best = (match.start(), choice)
    return best[1] if best else None


def extract_json(text: str) -> Any:
    """The first JSON value in ``text`` (code fences and leading prose are tolerated)."""
    body = text.strip()
    fence = _FENCE.match(body)
    if fence:
        body = fence.group(1).strip()
    try:
        return json.loads(body)
    except ValueError:
        pass
    decoder = json.JSONDecoder()
    for index, char in enumerate(body):
        if char in "{[":
            try:
                value, _end = decoder.raw_decode(body, index)
                return value
            except ValueError:
                continue
    raise ValueError("no JSON value found")


_JSON_TYPES = {"string": str, "integer": int, "number": (int, float), "boolean": bool, "array": list, "object": dict,
               "null": type(None)}


def _score_json(text: str, spec: Any) -> CaseScore:
    try:
        value = extract_json(text)
    except ValueError:
        return CaseScore(0.0, False, "output is not valid JSON")
    if not isinstance(spec, dict):
        ok = value == spec
        return _binary(ok, "JSON equals the expected value" if ok else "JSON differs from the expected value")
    checks: list[tuple[bool, str]] = [(True, "valid JSON")]
    if "type" in spec:
        checks.append((_is_type(value, spec["type"]), f"type {spec['type']}"))
    if "value" in spec:
        checks.append((value == spec["value"], "exact value"))
    obj = value if isinstance(value, dict) else {}
    for key in spec.get("required") or []:
        checks.append((key in obj, f"has '{key}'"))
    for key, kind in (spec.get("types") or {}).items():
        checks.append((key in obj and _is_type(obj[key], kind), f"'{key}' is {kind}"))
    for key, expected in (spec.get("equals") or {}).items():
        actual = obj.get(key)
        same = (isinstance(actual, str) and isinstance(expected, str) and actual.strip().lower() == expected.lower()) \
            or actual == expected
        checks.append((key in obj and same, f"'{key}' == {expected!r}"))
    for key, pattern in (spec.get("patterns") or {}).items():
        checks.append((key in obj and isinstance(obj[key], str) and re.search(pattern, obj[key]) is not None,
                       f"'{key}' matches {pattern}"))
    if "items" in spec:
        items = value if isinstance(value, list) else []
        lowered = [str(i).strip().lower() for i in items]
        for item in spec["items"]:
            checks.append((str(item).lower() in lowered, f"contains {item!r}"))
    passed = sum(1 for ok, _ in checks if ok)
    failed = [label for ok, label in checks if not ok]
    return CaseScore(passed / len(checks), not failed, "all checks passed" if not failed else "failed: " + ", ".join(failed))


def _is_type(value: Any, kind: str) -> bool:
    expected = _JSON_TYPES.get(kind)
    if expected is None:
        return False
    if kind in ("integer", "number") and isinstance(value, bool):
        return False
    if kind == "integer" and isinstance(value, float):
        return value.is_integer()
    return isinstance(value, expected)


def _binary(ok: bool, detail: str) -> CaseScore:
    return CaseScore(1.0 if ok else 0.0, ok, detail)


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return list(value) if isinstance(value, (list, tuple)) else [value]


def _normalize(text: str) -> str:
    text = re.sub(r"\s+", " ", text.strip().lower())
    return text.strip(" \"'`.!?,;:")


def _squash(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _show(value: Any) -> str:
    text = json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else repr(value)
    return text if len(text) <= 80 else text[:77] + "…"


# ---------------------------------------------------------------------------
# summaries
# ---------------------------------------------------------------------------


def summarize(results: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-case results of one model."""
    rows = list(results)
    scored = [r for r in rows if r.get("score") is not None and not r.get("error")]
    ttft = [float(r["ttft_ms"]) for r in rows if r.get("ttft_ms") is not None and not r.get("error")]
    tps = [float(r["tokens_per_second"]) for r in rows if r.get("tokens_per_second") and not r.get("error")]
    latency = [float(r["latency_ms"]) for r in rows if r.get("latency_ms") is not None and not r.get("error")]
    tokens = [int(r.get("completion_tokens") or 0) for r in rows if not r.get("error")]
    by_tag: dict[str, list[dict[str, Any]]] = {}
    for row in scored:
        for tag in row.get("tags") or []:
            by_tag.setdefault(tag, []).append(row)
    return {
        "cases": len(rows),
        "scored": len(scored),
        "passed": sum(1 for r in scored if r.get("passed")),
        "accuracy": (sum(1 for r in scored if r.get("passed")) / len(scored)) if scored else None,
        "mean_score": statistics.fmean(float(r["score"]) for r in scored) if scored else None,
        "errors": sum(1 for r in rows if r.get("error")),
        "ttft_ms_p50": _percentile(ttft, 50),
        "ttft_ms_p95": _percentile(ttft, 95),
        "tokens_per_second": statistics.median(tps) if tps else None,
        "latency_ms_p50": _percentile(latency, 50),
        "completion_tokens": sum(tokens),
        "by_tag": {tag: sum(1 for r in items if r.get("passed")) / len(items) for tag, items in sorted(by_tag.items())},
    }


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = (len(ordered) - 1) * pct / 100.0
    low, high = math.floor(rank), math.ceil(rank)
    if low == high:
        return ordered[int(rank)]
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


# ---------------------------------------------------------------------------
# thresholds (CI mode)
# ---------------------------------------------------------------------------

_THRESHOLD = re.compile(r"^\s*([a-z_0-9]+)\s*(>=|<=|>|<|==)\s*(-?[\d.]+)\s*$")
METRIC_ALIASES = {"accuracy": "accuracy", "acc": "accuracy", "score": "mean_score", "mean_score": "mean_score",
                  "tok_s": "tokens_per_second", "tokens_per_second": "tokens_per_second", "tps": "tokens_per_second",
                  "ttft_ms": "ttft_ms_p50", "ttft": "ttft_ms_p50", "ttft_ms_p50": "ttft_ms_p50",
                  "ttft_ms_p95": "ttft_ms_p95", "latency_ms": "latency_ms_p50", "errors": "errors"}


@dataclass(frozen=True)
class Threshold:
    metric: str
    op: str
    value: float

    def check(self, summary: dict[str, Any]) -> tuple[bool, str]:
        actual = summary.get(self.metric)
        if actual is None:
            return False, f"{self.metric} was not measured"
        ok = {">=": actual >= self.value, "<=": actual <= self.value, ">": actual > self.value,
              "<": actual < self.value, "==": actual == self.value}[self.op]
        return ok, f"{self.metric} = {actual:.4g} (needs {self.op} {self.value:g})"


def parse_threshold(text: str) -> Threshold:
    """``accuracy>=0.8``, ``tok_s>=20``, ``ttft_ms<=1000`` …"""
    match = _THRESHOLD.match(text)
    if not match:
        raise DatasetError(f"invalid threshold {text!r}; use e.g. accuracy>=0.8 or ttft_ms<=1000")
    name, op, value = match.groups()
    metric = METRIC_ALIASES.get(name)
    if metric is None:
        raise DatasetError(f"unknown metric '{name}' (use: {', '.join(sorted(METRIC_ALIASES))})")
    return Threshold(metric, op, float(value))


def check_thresholds(summary: dict[str, Any], thresholds: Iterable[Threshold]) -> list[str]:
    """Human-readable failures (empty when every threshold holds)."""
    failures = []
    for threshold in thresholds:
        ok, message = threshold.check(summary)
        if not ok:
            failures.append(message)
    return failures
