"""Adversarial Text-to-Speech (TTS) Test Sentence Generator.

Produces calibrated, domain-specific evaluation sentences specifically designed
to expose TTS acoustic model limitations (numerical formatting, code-switching,
acronyms, phonetic clusters, and honorific naming in Indonesian and English).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class TTSTestCase:
    id: str
    category: str
    language: str
    difficulty: str
    text: str
    expected_challenges: list[str]
    metadata: dict[str, Any]


_INDONESIAN_TEMPLATES: dict[str, list[dict[str, Any]]] = {
    "numbers": [
        {
            "text": "Pada tahun 2026, pendapatan meningkat sebesar Rp17.845.230,50 atau sekitar 13,72 persen.",
            "challenges": ["currency format", "decimal comma", "percentage", "large integer"],
            "difficulty": "hard",
        },
        {
            "text": "Saldo rekening nomor 0812-9988-7711 tercatat sebesar USD 4.520,75 pada tanggal 31 Desember 2025.",
            "challenges": ["hyphenated phone/account digits", "foreign currency prefix", "date format"],
            "difficulty": "medium",
        },
        {
            "text": "Suhu penyimpanan optimal adalah minus 18,5°C dengan toleransi ±0,25 derajat.",
            "challenges": ["negative numbers", "symbols", "fractions"],
            "difficulty": "hard",
        },
    ],
    "technical": [
        {
            "text": "PyTorch menggunakan CUDA 12.4 dengan tensor bertipe bfloat16 pada GPU NVIDIA RTX 4090.",
            "challenges": ["capitalized abbreviations", "mixed alphanumeric tokens", "technical terms"],
            "difficulty": "hard",
        },
        {
            "text": "Arsitektur Transformer mengandalkan Multi-Head Self-Attention dan feed-forward network secara bertingkat.",
            "challenges": ["machine learning nomenclature", "hyphenated English terms"],
            "difficulty": "medium",
        },
    ],
    "acronyms": [
        {
            "text": "API REST menggunakan OAuth 2.0 dan JWT untuk mengakses endpoint HTTPS pada cluster Kubernetes.",
            "challenges": ["all-caps acronyms", "version decimals", "case-sensitive protocols"],
            "difficulty": "hard",
        },
        {
            "text": "Kementerian Kominfo bersama BSSN merumuskan regulasi PSE, PDP, serta sertifikasi ISO/IEC 27001.",
            "challenges": ["Indonesian government acronyms", "slash notation", "standard numbering"],
            "difficulty": "hard",
        },
    ],
    "names": [
        {
            "text": "Dr. Raden Mas Soerjono Kartohadiprodjo menghadiri konferensi nasional di Keraton Ngayogyakarta Hadiningrat.",
            "challenges": ["Javanese aristocratic honorifics", "compound proper nouns", "archaic spelling"],
            "difficulty": "very_hard",
        },
        {
            "text": "Prof. Dr. Ir. Hj. Siti Nurhaliza, M.Sc., Ph.D. memimpin delegasi sains ke Universitas Gadjah Mada.",
            "challenges": ["multiple academic titles", "abbreviated prefixes", "post-nominal degrees"],
            "difficulty": "hard",
        },
    ],
    "code_switching": [
        {
            "text": "Model tersebut menggunakan Retrieval-Augmented Generation dengan hybrid search dan cross-encoder reranking.",
            "challenges": ["Indonesian sentence structure with English technical phrases", "phonetic switching"],
            "difficulty": "hard",
        },
        {
            "text": "Tolong push commit terbaru ke branch main lalu trigger workflow deployment di GitHub Actions.",
            "challenges": ["Indonesian conversational loanwords", "developer jargon"],
            "difficulty": "medium",
        },
    ],
    "phonetics": [
        {
            "text": "Konfigurasi tersebut memprioritaskan sinkronisasi, standardisasi, dan interoperabilitas sistemik.",
            "challenges": ["repetitive multi-syllable Latin prefixes", "tongue-twisting consonant clusters"],
            "difficulty": "very_hard",
        },
        {
            "text": "Kucing-kucing mencakar-cakar pagar bambu di belakang kandang kambing Pak Subur.",
            "challenges": ["reduplication phonetics", "plosive consonants"],
            "difficulty": "medium",
        },
    ],
    "long_form": [
        {
            "text": "Berdasarkan laporan komprehensif yang dirilis oleh Badan Kebijakan Fiskal pada kuartal ketiga tahun ini, stabilitas makroekonomi nasional tetap terjaga berkat sinergi kebijakan moneter yang responsif, surplus neraca perdagangan yang berkelanjutan, serta penguatan sektor manufaktur berorientasi ekspor.",
            "challenges": ["sustained breath control", "multi-clause cadence", "prosody consistency"],
            "difficulty": "hard",
        },
    ],
}


def generate_tts_suite(categories: list[str] | None = None, count: int = 10,
                       language: str = "id") -> list[TTSTestCase]:
    """Generate a calibrated suite of difficult TTS test cases."""
    selected_cats = categories or list(_INDONESIAN_TEMPLATES.keys())
    pool: list[dict[str, Any]] = []

    for cat in selected_cats:
        items = _INDONESIAN_TEMPLATES.get(cat, [])
        for item in items:
            pool.append({"category": cat, **item})

    if not pool:
        return []

    # Cycle or sample to satisfy requested count
    cases: list[TTSTestCase] = []
    idx = 1
    while len(cases) < count:
        for template in pool:
            cases.append(TTSTestCase(
                id=f"tts-test-{idx:03d}",
                category=template["category"],
                language=language,
                difficulty=template.get("difficulty", "medium"),
                text=template["text"],
                expected_challenges=template.get("challenges", []),
                metadata={"template_index": idx},
            ))
            idx += 1
            if len(cases) >= count:
                break

    return cases
