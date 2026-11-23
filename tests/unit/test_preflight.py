from __future__ import annotations

import pytest

from noelle.runtime.preflight import (
    GB,
    MB,
    MemoryBudget,
    assess_compatibility,
    context_limit,
    estimate_runtime_bytes,
    format_bytes,
    format_params,
    kv_cache_bytes_per_token,
    params_from_safetensors,
    parse_param_count,
    plan_queue,
    tune_generation,
    weights_bytes,
)

LLAMA_8B = {"num_hidden_layers": 32, "num_attention_heads": 32, "num_key_value_heads": 8, "hidden_size": 4096,
            "max_position_embeddings": 131072}


@pytest.mark.parametrize("name, expected", [
    ("Qwen/Qwen2.5-0.5B-Instruct", 0.5e9),
    ("meta-llama/Llama-3.1-8B-Instruct", 8e9),
    ("HuggingFaceTB/SmolLM2-1.7B", 1.7e9),
    ("HuggingFaceTB/SmolLM2-135M-Instruct", 135e6),
    ("google/gemma-2-9b-it", 9e9),
    ("mistralai/Mixtral-8x7B-v0.1", 56e9),
    ("Qwen/Qwen3-30B-A3B", 30e9),
    ("openai/whisper-large-v3", None),
    ("bert-base-uncased", None),
])
def test_parse_param_count(name: str, expected: float | None) -> None:
    result = parse_param_count(name)
    if expected is None:
        assert result is None
    else:
        assert result == pytest.approx(expected, rel=1e-6)


def test_params_from_safetensors_counts_bytes_per_dtype() -> None:
    count, size = params_from_safetensors({"BF16": 1000, "F32": 10, "I8": 4})
    assert count == 1014
    assert size == 1000 * 2 + 10 * 4 + 4


def test_weights_bytes_by_precision() -> None:
    assert weights_bytes(1_000_000, "fp16") == 2_000_000
    assert weights_bytes(1_000_000, "fp32") == 4_000_000
    assert weights_bytes(1_000_000, "int8") == 1_000_000
    assert weights_bytes(1_000_000, "nf4") < 600_000


def test_kv_cache_uses_gqa_heads() -> None:
    # 2 (K and V) * 32 layers * 8 kv heads * 128 head dim * 2 bytes
    assert kv_cache_bytes_per_token(LLAMA_8B) == 2 * 32 * 8 * 128 * 2
    assert context_limit(LLAMA_8B) == 131072
    nested = {"text_config": {"num_hidden_layers": 2, "num_attention_heads": 4, "hidden_size": 64}}
    assert kv_cache_bytes_per_token(nested) == 2 * 2 * 4 * 16 * 2


def test_compatibility_levels() -> None:
    budget = MemoryBudget(int(24 * GB), 0.15, int(64 * GB), "cuda")
    assert assess_compatibility(int(4e9), budget).level == "green"
    small = MemoryBudget(int(12 * GB), 0.15, int(32 * GB), "cuda")
    assert assess_compatibility(int(8e9), small, config=LLAMA_8B).level == "yellow"
    tiny = MemoryBudget(int(6 * GB), 0.15, int(16 * GB), "mps")
    red = assess_compatibility(int(30e9), tiny)
    assert red.level == "red"
    assert assess_compatibility(None, budget).level == "unknown"
    yellow = assess_compatibility(int(8e9), MemoryBudget(int(12 * GB), 0.15, int(16 * GB), "mps"), config=LLAMA_8B)
    assert yellow.recommended_precision == "mlx-4bit"


def test_estimate_includes_kv_and_overhead() -> None:
    est = estimate_runtime_bytes(int(8e9), "fp16", config=LLAMA_8B, context_tokens=4096)
    assert est > 16e9
    assert est < 17.5e9


def test_tune_generation_clamps_to_context_and_memory() -> None:
    by_ctx = tune_generation(4096, 2000, context_window=2048, kv_bytes_per_token=None, free_bytes=None)
    assert by_ctx.clamped and by_ctx.max_new_tokens == 48
    by_mem = tune_generation(4096, 100, context_window=None, kv_bytes_per_token=1 * MB, free_bytes=600 * MB)
    assert by_mem.clamped and by_mem.max_new_tokens == 500
    free = tune_generation(256, 10, context_window=8192, kv_bytes_per_token=1024, free_bytes=GB)
    assert not free.clamped and free.max_new_tokens == 256


def test_plan_queue_retain_when_everything_fits() -> None:
    plan = plan_queue([(1, "a", 2 * GB), (2, "b", 3 * GB)], usable_bytes=10 * GB)
    assert plan.mode == "retain"
    assert all(not step.evict_before for step in plan.steps)


def test_plan_queue_hot_swaps_in_view_order() -> None:
    jobs = [(2, "b", 8 * GB), (1, "a", 8 * GB), (3, "c", 8 * GB), (4, "d", 8 * GB)]
    plan = plan_queue(jobs, usable_bytes=12 * GB)
    assert plan.mode == "hot_swap"
    assert [s.view for s in plan.steps] == [1, 2, 3, 4]
    assert plan.steps[0].evict_before == []
    assert plan.steps[1].evict_before == ["a"]
    assert plan.steps[2].evict_before == ["b"]
    assert plan.steps[3].evict_before == ["c"]


def test_plan_queue_reuses_resident_and_prefers_unneeded_victims() -> None:
    plan = plan_queue([(1, "a", 4 * GB), (2, "b", 4 * GB), (3, "a", 4 * GB)], usable_bytes=9 * GB,
                      resident={"old": 4 * GB})
    assert plan.steps[0].evict_before == []
    assert plan.steps[1].evict_before == ["old"]
    assert plan.steps[2].reuse_loaded


def test_formatting() -> None:
    assert format_params(int(7.6e9)) == "7.6B"
    assert format_params(int(135e6)) == "135M"
    assert format_bytes(1536) == "2 KB"
    assert format_bytes(5 * GB) == "5.0 GB"
