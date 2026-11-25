"""Speech-to-Text (STT) Robustness Evaluation and Metric Computations.

Computes Word Error Rate (WER) and Character Error Rate (CER) via Levenshtein
distance, and synthesizes adversarial STT test cases.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Sequence


@dataclass
class STTMetricResult:
    wer: float
    cer: float
    substitutions: int
    deletions: int
    insertions: int
    word_count: int


def _levenshtein_distance(seq1: Sequence[Any], seq2: Sequence[Any]) -> tuple[int, int, int, int]:
    """Calculate edit distance and counts of substitutions, deletions, insertions."""
    n, m = len(seq1), len(seq2)
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],       # deletion
                    dp[i][j - 1],       # insertion
                    dp[i - 1][j - 1],   # substitution
                )

    # Backtrace to count op types
    i, j = n, m
    subs, dels, inss = 0, 0, 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and seq1[i - 1] == seq2[j - 1]:
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            subs += 1
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            dels += 1
            i -= 1
        else:
            inss += 1
            j -= 1

    return dp[n][m], subs, dels, inss


def normalize_transcript(text: str) -> str:
    """Standardize punctuation and whitespace for fair speech evaluation."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def calculate_wer(reference: str, hypothesis: str) -> STTMetricResult:
    """Calculate Word Error Rate (WER) between ground truth and ASR hypothesis."""
    ref_words = normalize_transcript(reference).split()
    hyp_words = normalize_transcript(hypothesis).split()

    if not ref_words:
        wer = 0.0 if not hyp_words else 1.0
        return STTMetricResult(wer, 0.0, 0, 0, len(hyp_words), 0)

    total_dist, subs, dels, inss = _levenshtein_distance(ref_words, hyp_words)
    wer = total_dist / float(len(ref_words))

    # Also compute Character Error Rate (CER)
    ref_chars = list(normalize_transcript(reference).replace(" ", ""))
    hyp_chars = list(normalize_transcript(hypothesis).replace(" ", ""))
    char_dist, _, _, _ = _levenshtein_distance(ref_chars, hyp_chars)
    cer = (char_dist / float(len(ref_chars))) if ref_chars else 0.0

    return STTMetricResult(
        wer=round(wer, 4),
        cer=round(cer, 4),
        substitutions=subs,
        deletions=dels,
        insertions=inss,
        word_count=len(ref_words),
    )
