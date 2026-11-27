"""Speculative-decoding configuration and compatibility checks.

Never enables speculation because two repositories have similar names. The draft and
target must share a tokenizer vocabulary; prompt-lookup needs no extra model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

MODES = ("off", "prompt_lookup", "draft")
DEFAULT_DRAFT_TOKENS = 10


@dataclass(frozen=True)
class SpeculativeConfig:
    mode: str = "off"
    draft_model: str | None = None
    draft_tokens: int = DEFAULT_DRAFT_TOKENS

    @property
    def enabled(self) -> bool:
        return self.mode in ("prompt_lookup", "draft")

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> SpeculativeConfig:
        data = data or {}
        mode = str(data.get("mode") or "off").lower()
        if mode not in MODES:
            mode = "off"
        return cls(mode=mode, draft_model=data.get("draft_model") or None,
                   draft_tokens=max(int(data.get("draft_tokens") or DEFAULT_DRAFT_TOKENS), 1))

    def to_dict(self) -> dict[str, Any]:
        return {"mode": self.mode, "draft_model": self.draft_model, "draft_tokens": self.draft_tokens}


def compatibility(target_config: Mapping[str, Any], draft_config: Mapping[str, Any], *,
                  target_tokenizer: Any = None, draft_tokenizer: Any = None) -> tuple[bool, str]:
    """Return ``(compatible, reason)``; reason is empty when compatible."""
    target_vocab = target_config.get("vocab_size")
    draft_vocab = draft_config.get("vocab_size")
    if target_vocab and draft_vocab and int(target_vocab) != int(draft_vocab):
        return False, f"Vocabulary sizes differ ({target_vocab} vs {draft_vocab})."
    if target_tokenizer is not None and draft_tokenizer is not None:
        if type(target_tokenizer).__name__ != type(draft_tokenizer).__name__:
            return False, "Tokenizers are different classes."
        target_len = len(target_tokenizer) if hasattr(target_tokenizer, "__len__") else None
        draft_len = len(draft_tokenizer) if hasattr(draft_tokenizer, "__len__") else None
        if target_len and draft_len and target_len != draft_len:
            return False, f"Tokenizer sizes differ ({target_len} vs {draft_len})."
    target_bos = target_config.get("bos_token_id")
    draft_bos = draft_config.get("bos_token_id")
    if target_bos is not None and draft_bos is not None and target_bos != draft_bos:
        return False, "BOS token ids differ."
    target_eos = target_config.get("eos_token_id")
    draft_eos = draft_config.get("eos_token_id")
    if target_eos is not None and draft_eos is not None and target_eos != draft_eos:
        return False, "EOS token ids differ."
    if not target_vocab and not draft_vocab and target_tokenizer is None:
        return False, "Not enough metadata to verify tokenizer compatibility."
    return True, ""


def suggest_drafts(target_model_id: str, records: Iterable[Any], *, limit: int = 5) -> list[str]:
    """Same-publisher, smaller causal-LM candidates (a hint, never an automatic choice)."""
    publisher = target_model_id.split("/")[0] if "/" in target_model_id else ""
    candidates: list[tuple[int, str]] = []
    for record in records:
        repo_id = getattr(record, "repo_id", "")
        if repo_id == target_model_id or getattr(record, "pipeline_tag", "") != "text-generation":
            continue
        if getattr(record, "source", "hub") == "quantized":
            continue
        same_publisher = repo_id.split("/")[0] == publisher if "/" in repo_id else False
        candidates.append((0 if same_publisher else 1, repo_id))
    candidates.sort()
    return [repo_id for _rank, repo_id in candidates[:limit]]