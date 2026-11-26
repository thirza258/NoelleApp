"""Unit tests for adversarial test generators, STT metrics, Embedding Lab, and Decision Advisor."""

from __future__ import annotations

from noelle.evaluation.generators.stt_generator import calculate_wer, normalize_transcript
from noelle.evaluation.generators.tts_generator import generate_tts_suite
from noelle.evaluation.labs.audio_ab_lab import AudioABLab
from noelle.evaluation.labs.embedding_lab import EmbeddingEvaluationLab
from noelle.services.cost_advisor import DecisionAdvisor, LocalHardwareParams


def test_tts_generator_categories() -> None:
    suite = generate_tts_suite(count=12)
    assert len(suite) == 12

    # Verify categories exist
    categories = {tc.category for tc in suite}
    assert "numbers" in categories
    assert "technical" in categories
    assert "acronyms" in categories

    # Verify Indonesian currency and acronym content
    num_cases = [tc for tc in suite if tc.category == "numbers"]
    assert any("Rp17.845.230,50" in tc.text for tc in num_cases)

    tech_cases = [tc for tc in suite if tc.category == "technical"]
    assert any("CUDA" in tc.text for tc in tech_cases)


def test_stt_wer_and_cer_metric() -> None:
    ref = "Pada tahun 2026 pendapatan meningkat pesat"
    # Exact match -> 0.0 WER
    res_exact = calculate_wer(ref, ref)
    assert res_exact.wer == 0.0
    assert res_exact.cer == 0.0

    # 1 substitution ("meningkat" -> "naik")
    hyp_sub = "Pada tahun 2026 pendapatan naik pesat"
    res_sub = calculate_wer(ref, hyp_sub)
    assert res_sub.substitutions == 1
    assert res_sub.wer > 0.0

    # Normalization handles punctuation and casing
    norm = normalize_transcript("Halo, Noelle?!  Ini... Uji Coba.")
    assert norm == "halo noelle ini uji coba"


def test_embedding_evaluation_lab(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from noelle.providers import local_provider as provider_module
    from noelle.providers.registry import get_provider

    keywords = ("privasi", "data", "ai", "rendang", "cuda", "gpu", "resep", "hardware")

    def vector(text: str) -> list[float]:
        values = [1.0 if keyword in text.lower() else 0.0 for keyword in keywords]
        if not any(values):
            values[0] = 0.1
        return values

    class FakeWorker:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def call(self, _cmd, payload, timeout=None):  # type: ignore[no-untyped-def]
            return {"embeddings": [vector(text) for text in payload["inputs"]], "dimension": len(keywords)}

    monkeypatch.setattr(provider_module, "WorkerClient", FakeWorker)
    local = get_provider("local")
    previous_endpoint = getattr(local, "_endpoint", None)
    previous_token = getattr(local, "_token", None)
    local._endpoint = "tcp://127.0.0.1:1"  # type: ignore[attr-defined]
    local._token = None  # type: ignore[attr-defined]
    try:
        _run_embedding_lab()
    finally:
        local._endpoint = previous_endpoint  # type: ignore[attr-defined]
        local._token = previous_token  # type: ignore[attr-defined]


def _run_embedding_lab() -> None:
    lab = EmbeddingEvaluationLab()
    docs = [
        {"id": "doc_1", "text": "Kecerdasan buatan lokal dan privasi data."},
        {"id": "doc_2", "text": "Resep membuat rendang daging sapi Padang."},
        {"id": "doc_3", "text": "Optimalisasi hardware GPU NVIDIA RTX 4090 dengan CUDA."},
    ]
    queries = [
        {"id": "q_1", "text": "privasi data model AI", "relevant_doc_ids": ["doc_1"]},
        {"id": "q_2", "text": "akselerasi CUDA grafis", "relevant_doc_ids": ["doc_3"]},
    ]
    lab.load_corpus(docs, queries)

    # Benchmark local provider
    res = lab.benchmark_model("local", "BAAI/bge-m3", k_values=(1, 2))
    assert res.provider_id == "local"
    assert res.model_id == "BAAI/bge-m3"
    assert res.total_cost_usd == 0.0
    assert 1 in res.recall_at_k
    assert res.avg_latency_ms >= 0.0


def test_audio_ab_lab() -> None:
    lab = AudioABLab()
    cand_a, cand_b = lab.synthesize_pair(
        text="Uji perbandingan audio Noelle",
        provider_a="local",
        model_a="SpeechT5",
        provider_b="local",
        model_b="Bark",
    )
    assert isinstance(cand_a.audio_bytes, bytes)
    assert isinstance(cand_b.audio_bytes, bytes)
    assert cand_a.cost_usd == 0.0
    assert cand_b.cost_usd == 0.0


def test_decision_advisor_tco_calculation() -> None:
    params = LocalHardwareParams(
        hardware_cost_usd=2400.0,
        amortization_months=24,
        power_draw_watts=450.0,
        electricity_kwh_cost=0.15,
        active_hours_per_day=8.0,
    )
    # Monthly amortization: $2400 / 24 = $100.00
    assert params.monthly_amortization_usd == 100.0

    # Daily kWh: (450/1000) * 8 = 3.6 kWh/day -> 3.6 * 30 * 0.15 = $16.20/month
    assert round(params.monthly_electricity_usd, 2) == 16.20

    advisor = DecisionAdvisor(params)
    # 500,000 queries with GPT-4o
    assessment = advisor.evaluate(projected_monthly_queries=500_000, avg_input_tokens=1200, avg_output_tokens=350, cloud_model="gpt-4o")

    assert assessment.cloud_monthly_cost_usd > 1000.0
    assert assessment.local_monthly_cost_usd == 116.20
    assert assessment.monthly_savings_usd > 0.0
    assert assessment.savings_percentage > 90.0
    assert assessment.recommended_strategy == "Dedicated Local Workstation"
    assert assessment.break_even_queries_per_day > 0
