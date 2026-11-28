"""Dataset Studio: load, normalize, validate, deduplicate, split and report local data.

All work is local. The report contains counts and short previews only; the training
manifest stores the dataset hash, not its contents.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

SUPPORTED_SUFFIXES = (".jsonl", ".json", ".csv", ".parquet", ".txt", ".md", ".markdown")
ROLES = ("system", "user", "assistant", "tool")


@dataclass
class DatasetReport:
    source: str
    sha256: str | None = None
    rows: int = 0
    valid: int = 0
    invalid: int = 0
    duplicates: int = 0
    empty_responses: int = 0
    very_long: int = 0
    avg_tokens: float = 0.0
    max_tokens: int = 0
    train: int = 0
    validation: int = 0
    schema: str = "unknown"
    preview: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def valid_ratio(self) -> float:
        return self.valid / self.rows if self.rows else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source, "sha256": self.sha256, "rows": self.rows, "valid": self.valid,
            "invalid": self.invalid, "duplicates": self.duplicates, "empty_responses": self.empty_responses,
            "very_long": self.very_long, "avg_tokens": round(self.avg_tokens, 1),
            "max_tokens": self.max_tokens, "train": self.train, "validation": self.validation,
            "schema": self.schema, "preview": self.preview, "errors": self.errors, "warnings": self.warnings,
            "valid_ratio": round(self.valid_ratio, 3),
        }


@dataclass
class PreparedDataset:
    train_path: Path
    validation_path: Path
    report_path: Path
    report: DatasetReport

    def to_dict(self) -> dict[str, Any]:
        return {"train_path": str(self.train_path), "validation_path": str(self.validation_path),
                "report_path": str(self.report_path), "report": self.report.to_dict()}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> tuple[list[Any], int]:
    rows: list[Any] = []
    malformed = 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except ValueError:
                malformed += 1
                rows.append({"text": line})
    return rows, malformed


def _read_json(path: Path) -> tuple[list[Any], int]:
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, ValueError):
        return [], 1
    if isinstance(data, list):
        return data, 0
    if isinstance(data, Mapping):
        for key in ("data", "rows", "examples", "records"):
            value = data.get(key)
            if isinstance(value, list):
                return value, 0
    return [data], 0


def _read_csv(path: Path) -> tuple[list[Any], int]:
    try:
        with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
            return list(csv.DictReader(handle)), 0
    except OSError:
        return [], 1


def _read_parquet(path: Path) -> tuple[list[Any], int]:
    try:
        import pyarrow.parquet as pq
    except ImportError:
        return [], 1
    try:
        return pq.read_table(path).to_pylist(), 0
    except Exception:  # noqa: BLE001 - optional parser; report malformed
        return [], 1


def _read_text(path: Path) -> tuple[list[Any], int]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return [], 1
    blocks = [block.strip() for block in text.replace("\r\n", "\n").split("\n\n") if block.strip()]
    return [{"text": block} for block in blocks], 0


def load_raw_rows(path: str | Path) -> tuple[list[Any], int]:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return _read_jsonl(path)
    if suffix == ".json":
        return _read_json(path)
    if suffix == ".csv":
        return _read_csv(path)
    if suffix == ".parquet":
        return _read_parquet(path)
    if suffix in (".txt", ".md", ".markdown"):
        return _read_text(path)
    raise ValueError(f"Unsupported dataset format '{suffix}'. Use JSONL, JSON, CSV, Parquet, TXT or Markdown.")


def _content(row: Mapping[str, Any], *names: str) -> str | None:
    for name in names:
        value = row.get(name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def normalize_row(row: Any) -> tuple[dict[str, Any] | None, str | None, str]:
    """Return ``(normalized_row, error, schema)`` where normalized is ChatML or ``{"text"}``."""
    if isinstance(row, str):
        text = row.strip()
        return ({"text": text}, None, "text") if text else (None, "empty row", "text")
    if not isinstance(row, Mapping):
        return None, "row is not an object", "unknown"

    messages = row.get("messages") or row.get("conversations")
    if isinstance(messages, list) and messages:
        cleaned: list[dict[str, str]] = []
        for message in messages:
            if not isinstance(message, Mapping):
                continue
            role = str(message.get("role") or message.get("from") or "").lower()
            content = message.get("content") or message.get("value")
            if role in ("human", "user"):
                role = "user"
            elif role in ("gpt", "assistant", "bot"):
                role = "assistant"
            elif role == "system":
                role = "system"
            else:
                continue
            if isinstance(content, str) and content.strip():
                cleaned.append({"role": role, "content": content.strip()})
        if not cleaned:
            return None, "messages contain no usable turns", "chat"
        if not any(message["role"] == "assistant" for message in cleaned):
            return None, "no assistant turn", "chat"
        result = {"messages": cleaned}
        for key in ("feedback", "rating", "source", "id"):
            if key in row:
                result[key] = row[key]
        return result, None, "chat"

    instruction = _content(row, "instruction", "prompt", "question", "query", "input_text")
    output = _content(row, "output", "response", "answer", "completion", "assistant")
    input_text = _content(row, "input", "context")
    if instruction and output:
        user = f"{instruction}\n\n{input_text}" if input_text and input_text != instruction else instruction
        result = {"messages": [{"role": "user", "content": user},
                               {"role": "assistant", "content": output}]}
        for key in ("feedback", "rating", "source", "id"):
            if key in row:
                result[key] = row[key]
        schema = "instruction" if "instruction" in row else ("query" if "query" in row else "prompt")
        return result, None, schema

    text = _content(row, "text", "content", "document")
    if text:
        return {"text": text}, None, "text"
    return None, "no recognizable instruction/response or text fields", "unknown"


def _row_text(row: Mapping[str, Any]) -> str:
    if isinstance(row.get("messages"), list):
        return "\n".join(str(message.get("content", "")) for message in row["messages"])
    return str(row.get("text") or "")


def _token_count(text: str, tokenizer: Callable[[str], Any] | None) -> int:
    if tokenizer is not None:
        try:
            encoded = tokenizer(text)
            if isinstance(encoded, Mapping) and "input_ids" in encoded:
                ids = encoded["input_ids"]
                return len(ids[0]) if isinstance(ids, list) and ids and isinstance(ids[0], list) else len(ids)
        except Exception:  # noqa: BLE001 - fall back to a rough estimate
            pass
    return max(len(text) // 4, 0)


def analyze_dataset(path: str | Path, *, tokenizer: Callable[[str], Any] | None = None,
                    max_seq_length: int = 1024, validation_ratio: float = 0.1) -> DatasetReport:
    source = Path(path)
    raw, malformed = load_raw_rows(source)
    report = DatasetReport(source=str(source), sha256=file_sha256(source) if source.is_file() else None)
    seen: set[str] = set()
    schema_counts: dict[str, int] = {}
    lengths: list[int] = []
    preview: list[dict[str, Any]] = []
    for index, raw_row in enumerate(raw):
        report.rows += 1
        normalized, error, schema = normalize_row(raw_row)
        schema_counts[schema] = schema_counts.get(schema, 0) + 1
        if error or normalized is None:
            report.invalid += 1
            if len(report.errors) < 8:
                report.errors.append(f"row {index + 1}: {error}")
            continue
        text = _row_text(normalized)
        key = hashlib.sha1(json.dumps(normalized, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        if key in seen:
            report.duplicates += 1
            continue
        seen.add(key)
        if isinstance(normalized.get("messages"), list) and not any(
                message.get("role") == "assistant" and message.get("content") for message in normalized["messages"]):
            report.empty_responses += 1
            report.invalid += 1
            continue
        tokens = _token_count(text, tokenizer)
        lengths.append(tokens)
        if tokens > max_seq_length:
            report.very_long += 1
        report.valid += 1
        if len(preview) < 3:
            preview.append({"schema": schema, "tokens": tokens,
                            "preview": text[:240] + ("…" if len(text) > 240 else "")})
    if malformed:
        report.warnings.append(f"{malformed} malformed JSON line(s) were treated as plain text")
    report.schema = max(schema_counts, key=schema_counts.get) if schema_counts else "unknown"
    report.avg_tokens = sum(lengths) / len(lengths) if lengths else 0.0
    report.max_tokens = max(lengths) if lengths else 0
    report.preview = preview
    validation = int(round(report.valid * min(max(validation_ratio, 0.0), 0.5))) if report.valid > 1 else 0
    report.validation = min(validation, max(report.valid - 1, 0))
    report.train = report.valid - report.validation
    if report.valid == 0:
        report.errors.append("No usable rows after normalization.")
    if report.very_long:
        report.warnings.append(f"{report.very_long} sample(s) exceed {max_seq_length} tokens.")
    return report


def prepare_dataset(path: str | Path, output_dir: str | Path, *, tokenizer: Callable[[str], Any] | None = None,
                    max_seq_length: int = 1024, validation_ratio: float = 0.1, dedupe: bool = True,
                    drop_invalid: bool = True, trim_long: bool = False, seed: int = 42) -> PreparedDataset:
    source = Path(path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    raw, _malformed = load_raw_rows(source)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    invalid = 0
    for raw_row in raw:
        normalized, error, _schema = normalize_row(raw_row)
        if error or normalized is None:
            invalid += 1
            if drop_invalid:
                continue
            continue
        key = hashlib.sha1(json.dumps(normalized, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        if dedupe and key in seen:
            continue
        seen.add(key)
        if trim_long:
            text = _row_text(normalized)
            if _token_count(text, tokenizer) > max_seq_length:
                if isinstance(normalized.get("messages"), list):
                    trimmed: list[dict[str, str]] = []
                    for message in normalized["messages"]:
                        content = str(message.get("content") or "")
                        trimmed.append({**message, "content": content[: max(max_seq_length * 4, 200)]})
                    normalized = {**normalized, "messages": trimmed}
                else:
                    normalized = {**normalized, "text": str(normalized.get("text") or "")[: max_seq_length * 4]}
        rows.append(normalized)
    if not rows:
        raise ValueError("The dataset has no valid rows after normalization.")
    rng = random.Random(seed)
    rng.shuffle(rows)
    validation_count = min(int(round(len(rows) * min(max(validation_ratio, 0.0), 0.5))), max(len(rows) - 1, 0))
    validation_rows = rows[:validation_count]
    train_rows = rows[validation_count:]

    def write(target: Path, items: Iterable[dict[str, Any]]) -> None:
        with target.open("w", encoding="utf-8") as handle:
            for item in items:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    train_path = output / "train.jsonl"
    validation_path = output / "validation.jsonl"
    write(train_path, train_rows)
    write(validation_path, validation_rows)
    report = analyze_dataset(source, tokenizer=tokenizer, max_seq_length=max_seq_length,
                             validation_ratio=validation_ratio)
    report.train = len(train_rows)
    report.validation = len(validation_rows)
    report.invalid = max(report.invalid, invalid)
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    return PreparedDataset(train_path, validation_path, report_path, report)


def from_feedback(records: Iterable[Mapping[str, Any]], *, min_rating: float = 4.0,
                  positive_only: bool = True) -> list[dict[str, Any]]:
    """Convert feedback records into SFT rows; never treats unverified synthetic text as human data."""
    output: list[dict[str, Any]] = []
    for record in records:
        rating = record.get("rating")
        feedback = str(record.get("feedback") or "").lower()
        if positive_only:
            if isinstance(rating, (int, float)) and rating < min_rating:
                continue
            if feedback and feedback not in ("good", "positive", "up", "thumbs_up"):
                continue
        prompt = record.get("prompt") or record.get("query") or record.get("instruction")
        response = record.get("response") or record.get("output") or record.get("correction")
        if not isinstance(prompt, str) or not isinstance(response, str) or not prompt.strip() or not response.strip():
            continue
        output.append({"messages": [{"role": "user", "content": prompt.strip()},
                                    {"role": "assistant", "content": response.strip()}],
                       "source": "feedback", "rating": rating, "feedback": feedback or None})
    return output