"""Capability registry + runtime/quantization advisor (pure Python, no hardware needed)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from noelle.runtime import advisor, capabilities
from noelle.runtime.preflight import GB, MemoryBudget

QWEN7_CFG = {"architectures": ["Qwen2ForCausalLM"], "model_type": "qwen2", "num_hidden_layers": 28,
             "num_attention_heads": 28, "num_key_value_heads": 4, "hidden_size": 3584,
             "max_position_embeddings": 32768}
QWEN05_CFG = {**QWEN7_CFG, "num_hidden_layers": 24, "num_attention_heads": 14, "num_key_value_heads": 2,
              "hidden_size": 896}


def model(**kw: Any) -> SimpleNamespace:
    base = dict(id="org/model", pipeline_tag="text-generation", library_name="transformers", tags=[], siblings=[],
                config={}, param_count=None, weight_bytes=None, license="apache-2.0", gated=False, card_data={},
                gguf=None, base_models=[], is_adapter=False)
    base.update(kw)
    return SimpleNamespace(**base)


def mac(usable_gb: float = 12, **backends: bool) -> advisor.HardwareProfile:
    avail = {"transformers": True, "mlx": True, "llama.cpp": False, "llama.cpp-tools": False}
    avail.update(backends)
    return advisor.HardwareProfile("mps", "Apple M2 Pro GPU", MemoryBudget(int(usable_gb * GB), 0.15, int(16 * GB), "mps"),
                                   avail, advisor.memory_bandwidth("Apple M2 Pro GPU", "mps"))


def rtx3060(**backends: bool) -> advisor.HardwareProfile:
    avail = {"transformers": True, "bitsandbytes": True}
    avail.update(backends)
    name = "NVIDIA GeForce RTX 3060"
    return advisor.HardwareProfile("cuda", name, MemoryBudget(int(12 * GB), 0.15, int(32 * GB), "cuda"), avail,
                                   advisor.memory_bandwidth(name, "cuda"))


# ------------------------------------------------------------------ capabilities
def test_text_model_capabilities() -> None:
    cap = capabilities.analyze(model(
        id="Qwen/Qwen2.5-7B-Instruct", param_count=7_600_000_000, config=QWEN7_CFG,
        siblings=[{"name": "model-00001-of-00002.safetensors", "size": 1}, {"name": "config.json", "size": 1}],
        card_data={"language": ["en", "id"]}))
    assert cap.family == "text" and cap.inputs == ("text",) and cap.outputs == ("text",)
    assert cap.formats == ("safetensors",) and cap.quantization is None
    assert cap.runtimes == ("transformers",)
    assert "nf4" in cap.precisions and "mlx-4bit" in cap.conversions and "gguf-q4_k_m" in cap.conversions
    assert cap.context_length == 32768 and cap.architecture == "Qwen2ForCausalLM"
    assert cap.languages == ("en", "id") and cap.license == "apache-2.0" and not cap.remote_code.required
    assert cap.chat_template is None and cap.tool_calling is None  # unknown until a tokenizer config is seen
    data = cap.to_dict()
    assert data["supported"] and data["publisher"] == "Qwen" and data["modalities"] == ["text"]


def test_tool_calling_comes_from_the_chat_template() -> None:
    with_tools = capabilities.analyze(model(), tokenizer_config={"chat_template": "{% if tools %}{{ tools }}{% endif %}"})
    assert with_tools.chat_template is True and with_tools.tool_calling is True
    plain = capabilities.analyze(model(), tokenizer_config={"chat_template": "{{ messages }}"})
    assert plain.chat_template is True and plain.tool_calling is False
    none = capabilities.analyze(model(), tokenizer_config={"eos_token": "x"})
    assert none.chat_template is False and none.tool_calling is None
    tagged = capabilities.analyze(model(tags=["function-calling"]))
    assert tagged.tool_calling is True


def test_remote_code_lists_python_files_and_auto_map() -> None:
    cap = capabilities.analyze(model(tags=["custom_code"], config={"auto_map": {"AutoModel": "modeling_x.XModel"}},
                                     siblings=[{"name": "modeling_x.py"}, {"name": "configuration_x.py"},
                                               {"name": "model.safetensors"}]))
    assert cap.remote_code.required
    assert cap.remote_code.files == ("configuration_x.py", "modeling_x.py")
    assert cap.remote_code.auto_map == ("modeling_x.XModel",)


def test_gguf_repo_groups_variants_and_shards() -> None:
    cap = capabilities.analyze(model(
        id="bartowski/X-GGUF", library_name="gguf", tags=["gguf"], param_count=7_000_000_000,
        gguf={"context_length": 32768, "architecture": "qwen2"},
        siblings=[{"name": "X-Q4_K_M.gguf", "size": 4_000}, {"name": "X-Q8_0.gguf", "size": 8_000},
                  {"name": "X-f16-00001-of-00002.gguf", "size": 5}, {"name": "X-f16-00002-of-00002.gguf", "size": 6},
                  {"name": "mmproj-X-f16.gguf", "size": 1}, {"name": "README.md", "size": 1}]))
    assert cap.quantization == "gguf" and cap.runtimes == ("llama.cpp",) and cap.precisions == ("auto",)
    quants = {v.quant: v for v in cap.gguf_variants}
    assert set(quants) == {"Q4_K_M", "Q8_0", "F16"}  # the vision projector is not a variant
    assert quants["F16"].size_bytes == 11 and len(quants["F16"].files) == 2
    assert cap.context_length == 32768 and cap.architecture == "qwen2"


@pytest.mark.parametrize("name, quant", [
    ("llama-2-7b.Q4_K_M.gguf", "Q4_K_M"), ("Qwen3-8B-UD-Q4_K_XL.gguf", "Q4_K_XL"), ("m-IQ4_XS.gguf", "IQ4_XS"),
    ("m-q8_0-00001-of-00003.gguf", "Q8_0"), ("m-fp16.gguf", "F16"), ("model.gguf", "GGUF"),
])
def test_gguf_quant_labels(name: str, quant: str) -> None:
    (variant,) = capabilities.gguf_variants_from_files([name], {name: 10})
    assert variant.quant == quant


def test_prequantized_formats() -> None:
    mlx = capabilities.analyze(model(library_name="mlx", config={"quantization": {"bits": 4, "group_size": 64}}))
    assert mlx.quantization == "mlx" and mlx.runtimes == ("mlx",) and not mlx.conversions
    awq = capabilities.analyze(model(config={"quantization_config": {"quant_method": "awq"}}))
    assert awq.quantization == "awq" and awq.runtimes == ("transformers",)
    exl2 = capabilities.analyze(model(tags=["exl2"]))
    assert exl2.quantization == "exl2" and not exl2.supported


def test_unsupported_task_has_no_runtime() -> None:
    cap = capabilities.analyze(model(pipeline_tag="reinforcement-learning"))
    assert not cap.supported and cap.family is None and not cap.conversions


def test_detect_backends_never_imports(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    before = set(sys.modules)
    found = capabilities.detect_backends()
    assert {"transformers", "mlx", "llama.cpp", "bitsandbytes"} <= set(found)
    assert not {"torch", "mlx.core", "llama_cpp"} & (set(sys.modules) - before)


# ---------------------------------------------------------------------- advisor
def test_seven_b_on_a_16gb_mac_recommends_a_4bit_mlx_build() -> None:
    cap = capabilities.analyze(model(param_count=7_600_000_000, config=QWEN7_CFG))
    a = advisor.assess(cap, mac(), config=QWEN7_CFG)
    options = {o.key: o for o in a.options}
    assert options["fp16"].fits is False and options["fp16"].status == advisor.READY
    assert options["nf4"].status == advisor.UNSUPPORTED  # bitsandbytes is CUDA-only
    assert options["mlx-4bit"].status == advisor.BUILD and options["mlx-4bit"].fits
    assert options["gguf-q4_k_m"].status == advisor.INSTALL  # llama.cpp tools not installed
    assert a.recommended.key == "mlx-4bit" and a.recommended.action == "quantize"
    assert a.score == sum(item.points for item in a.items) and 60 <= a.score <= 90
    assert a.context_tokens == 32768
    assert "estimate" in a.speed_text()
    data = a.to_dict()
    assert data["recommended"]["key"] == "mlx-4bit" and len(data["breakdown"]) == 5


def test_local_build_makes_the_conversion_ready() -> None:
    cap = capabilities.analyze(model(param_count=7_600_000_000, config=QWEN7_CFG))
    a = advisor.assess(cap, mac(), config=QWEN7_CFG, local_variants={"mlx-4bit": "local/qwen-mlx"})
    assert a.recommended.key == "mlx-4bit" and a.recommended.status == advisor.READY
    assert a.recommended.action == "use" and a.recommended.local_variant == "local/qwen-mlx"


def test_small_model_runs_at_full_precision_and_scores_high() -> None:
    cap = capabilities.analyze(model(param_count=494_000_000, config=QWEN05_CFG))
    a = advisor.assess(cap, mac(), config=QWEN05_CFG)
    assert a.recommended.key == "fp16" and a.score == 100 and a.grade == "Excellent" and a.level == "green"
    fast = advisor.recommend(a.options, "fast", param_count=cap.param_count, device="mps", usable_bytes=int(10 * GB))
    assert fast.key == "mlx-4bit"  # the fast profile accepts a one-time local build


def test_cuda_card_uses_bitsandbytes() -> None:
    cap = capabilities.analyze(model(param_count=7_600_000_000, config=QWEN7_CFG))
    a = advisor.assess(cap, rtx3060(), config=QWEN7_CFG)
    options = {o.key: o for o in a.options}
    assert options["int8"].status == advisor.READY and options["mlx-4bit"].status == advisor.UNSUPPORTED
    assert a.recommended.key in ("nf4", "int8")
    assert advisor.recommend(a.options, "quality", param_count=cap.param_count, device="cuda",
                             usable_bytes=a.hardware.usable_bytes).key == "int8"


def test_prequantized_awq_on_a_mac_is_capped() -> None:
    cap = capabilities.analyze(model(param_count=7_000_000_000, weight_bytes=int(4.5 * GB),
                                     config={"quantization_config": {"quant_method": "awq"}}))
    a = advisor.assess(cap, mac())
    assert a.options[0].status == advisor.UNSUPPORTED
    assert a.score <= 15 and "cannot run on this device" in a.cap_note


def test_gguf_profiles() -> None:
    cap = capabilities.analyze(model(
        library_name="gguf", tags=["gguf"], param_count=7_600_000_000,
        siblings=[{"name": "X-Q4_K_M.gguf", "size": int(4.7e9)}, {"name": "X-Q8_0.gguf", "size": int(8.1e9)},
                  {"name": "X-IQ2_XS.gguf", "size": int(2.4e9)}]))
    hw = mac(**{"llama.cpp": True})
    a = advisor.assess(cap, hw)
    assert a.recommended.key == "gguf:Q4_K_M" and a.recommended.gguf_file == "X-Q4_K_M.gguf"
    pick = lambda p: advisor.recommend(a.options, p, param_count=cap.param_count, device="mps",  # noqa: E731
                                       usable_bytes=hw.usable_bytes).key
    assert pick("quality") == "gguf:Q8_0" and pick("low_memory") == "gguf:IQ2_XS"
    without = advisor.assess(cap, mac())
    assert without.recommended.status == advisor.INSTALL and without.score < a.score


def test_too_large_model_is_capped() -> None:
    cap = capabilities.analyze(model(param_count=405_000_000_000))
    a = advisor.assess(cap, mac())
    assert a.recommended is None and a.score <= 45 and a.cap_note


def test_trust_and_access_points() -> None:
    risky = capabilities.analyze(model(param_count=494_000_000, config=QWEN05_CFG, tags=["custom_code"], gated="auto",
                                       license=None))
    item = {i.key: i for i in advisor.assess(risky, mac(), config=QWEN05_CFG).items}["trust"]
    assert item.points == 0 and "repository code" in item.note
    trusted = {i.key: i for i in advisor.assess(risky, mac(), config=QWEN05_CFG, trusted=True).items}["trust"]
    assert trusted.points == 9


def test_measured_speed_wins_over_estimates() -> None:
    rows = [{"precision": "fp16", "tokens_per_second": 30.0, "ttft_ms": 400, "status": "finished",
             "metrics": {"engine": "transformers", "device": "mps"}},
            {"precision": "fp16", "tokens_per_second": 34.0, "ttft_ms": 380, "status": "finished", "metrics": {}},
            {"precision": "fp16", "tokens_per_second": 99.0, "status": "error"}]
    measured = advisor.summarize_measurements(rows)
    assert measured["fp16"].runs == 2 and measured["fp16"].tokens_per_second == 32.0
    cap = capabilities.analyze(model(param_count=494_000_000, config=QWEN05_CFG))
    a = advisor.assess(cap, mac(), config=QWEN05_CFG, measured=measured)
    assert a.speed_text().startswith("32.0 tok/s measured")


def test_auto_precision_runs_count_for_the_full_precision_option() -> None:
    measured = advisor.summarize_measurements([{"precision": "auto", "tokens_per_second": 98.8, "ttft_ms": 33,
                                                "status": "finished", "metrics": {}}])
    cap = capabilities.analyze(model(param_count=362_000_000, config=QWEN05_CFG))
    a = advisor.assess(cap, mac(), config=QWEN05_CFG, measured=measured)
    assert a.recommended.key == "fp16" and a.measured_for_recommended.tokens_per_second == 98.8
    mlx = next(o for o in a.options if o.key == "mlx-4bit")
    assert a.measured_for(mlx) is None  # an MLX build is a different engine: no borrowed numbers


def test_context_that_fits() -> None:
    cap = capabilities.analyze(model(param_count=7_600_000_000, config={**QWEN7_CFG, "max_position_embeddings": 131072,
                                                                        "num_key_value_heads": 28}))
    a = advisor.assess(cap, mac(), config={**QWEN7_CFG, "max_position_embeddings": 131072, "num_key_value_heads": 28})
    assert a.context_tokens is not None and a.context_tokens < 131072
    assert any("limits the context" in note for note in a.notes)
    unknown = capabilities.analyze(model(param_count=494_000_000))
    assert advisor.assess(unknown, mac()).context_tokens is None  # no config, no guess


@pytest.mark.parametrize("name, device, expected", [
    ("NVIDIA GeForce RTX 3060", "cuda", 360), ("NVIDIA GeForce RTX 3060 Ti", "cuda", 448),
    ("Apple M2 GPU", "mps", 100), ("Apple M2 Pro GPU", "mps", 200), ("Apple M2 Max GPU", "mps", 400),
    ("Mystery GPU", "cuda", None), ("Apple M2 Pro GPU", "cpu", None),
])
def test_memory_bandwidth_lookup(name: str, device: str, expected: float | None) -> None:
    assert advisor.memory_bandwidth(name, device) == expected


def test_compat_carries_score_for_badges() -> None:
    cap = capabilities.analyze(model(param_count=494_000_000, config=QWEN05_CFG))
    a = advisor.assess(cap, mac(), config=QWEN05_CFG)
    compat = advisor.with_score(a.compatibility, a)
    assert compat.score == 100 and compat.grade == "Excellent" and compat.level == "green"


def test_trust_applies_to_repos_publishers_and_local_builds() -> None:
    from noelle.app.services.model_registry import ModelRecord, remote_code_trusted

    settings = {"runtime.trusted_remote_code": ["org/one"], "runtime.trusted_publishers": ["acme"]}

    class S:
        def get(self, key: str, default: Any = None) -> Any:
            return settings.get(key, default)

    assert remote_code_trusted(S(), "org/one") and remote_code_trusted(S(), "acme/anything")
    assert not remote_code_trusted(S(), "org/two") and not remote_code_trusted(S(), "acmeish/x")
    build = ModelRecord(repo_id="local/one-mlx-4bit", base_model="org/one", source="quantized")
    assert build.worker_payload(S())["trust_remote_code"] is True  # a local build inherits its base's trust
    assert build.worker_payload()["trust_remote_code"] is False  # without settings: only the stored flag
    assert ModelRecord(repo_id="org/two").worker_payload(S())["trust_remote_code"] is False
