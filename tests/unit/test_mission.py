"""Mission planner: goal parsing, stated hardware, candidate ranking and the step list (fake Hub search)."""

from __future__ import annotations

from typing import Any

import pytest

from noelle.app.services import mission
from noelle.app.services.hf_client import ModelSummary
from noelle.runtime import advisor
from noelle.runtime.preflight import GB, MemoryBudget

CFG7 = {"architectures": ["Qwen2ForCausalLM"], "num_hidden_layers": 28, "num_attention_heads": 28,
        "num_key_value_heads": 4, "hidden_size": 3584, "max_position_embeddings": 32768}


def mac(**backends: bool) -> advisor.HardwareProfile:
    avail = {"transformers": True, "mlx": True, "sentence-transformers": True}
    avail.update(backends)
    return advisor.HardwareProfile("mps", "Apple M2 Pro GPU", MemoryBudget(int(12 * GB), 0.15, int(16 * GB), "mps"),
                                   avail, 200)


def llm(repo: str, params: float, *, langs: tuple[str, ...] = (), downloads: int = 100_000, tags: tuple[str, ...] = (),
        license: str = "apache-2.0") -> ModelSummary:
    return ModelSummary(id=repo, author=repo.split("/")[0], pipeline_tag="text-generation",
                        library_name="transformers", tags=["conversational", *tags], license=license,
                        param_count=int(params), downloads=downloads, config=CFG7,
                        card_data={"language": list(langs)} if langs else {})


POOL = [
    llm("Qwen/Qwen2.5-7B-Instruct", 7.6e9, downloads=5_000_000),
    llm("GoToCompany/llama3-8b-cpt-sahabatai-v1-instruct", 8e9, langs=("id", "jv", "su"), downloads=20_000,
        license="llama3"),
    llm("meta-llama/Llama-3.3-70B-Instruct", 70e9, downloads=3_000_000, license="llama3.3"),
    llm("org/SomeModel-7B", 7e9, downloads=9_000_000),  # base model: not instruction-tuned
    llm("Qwen/Qwen2.5-Coder-7B-Instruct", 7.6e9, downloads=1_000_000),
    llm("HuggingFaceTB/SmolLM2-135M-Instruct", 1.35e8, downloads=2_000_000),
]
POOL[3].tags = []  # neither "conversational" nor an instruct-style name


def embedder(repo: str) -> ModelSummary:
    return ModelSummary(id=repo, pipeline_tag="sentence-similarity", library_name="sentence-transformers",
                        param_count=568_000_000, license="mit", tags=["sentence-transformers"])


def fake_search(pool: list[ModelSummary]) -> Any:
    calls: list[dict[str, Any]] = []

    def search(**query: Any) -> list[ModelSummary]:
        calls.append(query)
        return list(pool)

    search.calls = calls  # type: ignore[attr-defined]
    return search


# ------------------------------------------------------------------------- goal
@pytest.mark.parametrize("text, task", [
    ("A local Indonesian legal assistant for my PDF contracts", "rag"),
    ("Chat with my notes", "rag"),
    ("A fast coding assistant for Python", "code"),
    ("Transcribe my meeting recordings", "asr"),
    ("Generate product images for my shop", "image"),
    ("Read text from photos of receipts", "vision"),
    ("Translate customer emails into English", "translation"),
    ("A friendly chatbot", "chat"),
    ("Saya butuh asisten untuk dokumen hukum", "rag"),
])
def test_task_detection(text: str, task: str) -> None:
    assert mission.parse_goal(text).task == task


def test_languages_profile_and_licence() -> None:
    goal = mission.parse_goal("A fast Indonesian and English assistant for a commercial product")
    assert goal.languages == ["id", "en"] and goal.profile == "fast" and goal.commercial
    assert mission.parse_goal("Saya ingin membuat aplikasi untuk tanya jawab").languages == ["id"]
    assert mission.parse_goal("best quality summaries").profile == "quality"
    assert mission.parse_goal("questions over long documents").profile == "max_context"
    assert mission.parse_goal("a coding helper on my laptop").profile == "balanced"  # a laptop is not a constraint


def test_stated_hardware() -> None:
    hw, note = mission.target_hardware("I have 16 GB RAM and an RTX 3060", mac())
    assert hw.device == "cuda" and hw.budget.total_bytes == 12 * GB and "RTX 3060" in note
    assert hw.has("bitsandbytes") and hw.bandwidth_gbps == 360
    hw, _ = mission.target_hardware("a GPU with 8GB VRAM", mac())
    assert hw.device == "cuda" and hw.budget.total_bytes == 8 * GB
    hw, note = mission.target_hardware("MacBook Air M2 with 24 GB", mac())
    assert hw.device == "mps" and hw.budget.total_bytes == int(24 * GB * 0.75) and "M2" in note
    detected = mac()
    assert mission.target_hardware("just a chatbot", detected) == (detected, None)
    assert "GPU with 8 GB" not in mission.GPU_VRAM_GB  # stated sizes never leak into the shared table


# ------------------------------------------------------------------------- plan
def test_indonesian_document_qa_on_an_rtx_3060() -> None:
    search = fake_search(POOL)
    planner = mission.Planner(search=search, info=embedder, hardware=mac())
    plan = planner.plan("I need a local model for Indonesian document QA. I have 16 GB RAM and an RTX 3060.")
    assert plan.goal.task == "rag" and plan.goal.languages == ["id"] and plan.hardware.device == "cuda"
    llm_choice = plan.component("llm").candidate
    assert llm_choice is not None and llm_choice.model_id != "meta-llama/Llama-3.3-70B-Instruct"  # does not fit
    assert llm_choice.model_id != "org/SomeModel-7B"  # base models are not assistants
    assert llm_choice.assessment.recommended.key in ("nf4", "int8", "fp16")
    assert any("id" in r or "Multilingual" in r for r in llm_choice.reasons)
    assert plan.component("embedding").candidate.model_id == "BAAI/bge-m3"
    assert [s.key for s in plan.steps] == ["download", "download", "knowledge", "test", "serve"]
    assert all(s.approval for s in plan.steps if s.key in ("download", "serve"))
    assert plan.eval_dataset == "indonesian-qa"
    assert any("Planned for the hardware you described" in w for w in plan.warnings)
    assert any(q.get("filter") == "id" for q in search.calls)
    data = plan.to_dict()
    assert data["hardware"]["device"] == "cuda" and data["components"][0]["choice"]["model_id"]
    assert "## Recommended stack" in plan.to_markdown()


def test_code_goal_prefers_a_coder_model() -> None:
    plan = mission.Planner(search=fake_search(POOL), info=None, hardware=mac()).plan("A coding assistant")
    assert "Coder" in plan.component("llm").candidate.model_id
    assert plan.eval_dataset == "coding-python" and plan.steps[-1].title == "Serve as 'noelle-code'"


def test_commercial_goal_penalises_restrictive_licences() -> None:
    plan = mission.Planner(search=fake_search(POOL), info=None, hardware=mac()).plan(
        "Indonesian chatbot for a commercial product, I have an RTX 4090")
    choice = plan.component("llm").candidate
    assert choice.assessment.capability.license == "apache-2.0"


def test_downloaded_model_skips_the_download(tmp_path: Any) -> None:
    from noelle.app.services.model_registry import ModelRecord

    local = ModelRecord(repo_id="Qwen/Qwen2.5-1.5B-Instruct", pipeline_tag="text-generation", local_path=str(tmp_path),
                        param_count=1_540_000_000, license="apache-2.0")
    offline = fake_search([])
    plan = mission.Planner(search=offline, info=None, hardware=mac(), local_models=[local]).plan("A chat assistant")
    choice = plan.component("llm").candidate
    assert choice.model_id == "Qwen/Qwen2.5-1.5B-Instruct" and choice.downloaded
    assert "Already downloaded" in choice.reasons and plan.steps[0].key != "download"


def test_offline_with_nothing_local_warns() -> None:
    def broken(**_q: Any) -> list[ModelSummary]:
        raise OSError("offline")

    plan = mission.Planner(search=broken, info=None, hardware=mac()).plan("A chat assistant")
    assert plan.component("llm").candidate is None and plan.warnings and not plan.steps


def test_build_and_install_steps() -> None:
    big = [llm("Qwen/Qwen2.5-14B-Instruct", 14.7e9, downloads=1_000_000)]
    plan = mission.Planner(search=fake_search(big), info=None, hardware=mac()).plan("A chat assistant")
    keys = [s.key for s in plan.steps]
    assert keys[:2] == ["download", "build"] and plan.steps[1].approval  # 4-bit MLX build on a 16 GB Mac
    plan = mission.Planner(search=fake_search(big), info=None, hardware=mac(mlx=False)).plan("A chat assistant")
    assert "install" in [s.key for s in plan.steps] or plan.component("llm").candidate is None


def test_speech_goal_prefers_whisper() -> None:
    pool = [ModelSummary(id="new/fancy-asr", pipeline_tag="automatic-speech-recognition", param_count=2e9,
                         downloads=9_000_000, tags=["custom_code"], license="apache-2.0"),
            ModelSummary(id="openai/whisper-large-v3-turbo", pipeline_tag="automatic-speech-recognition",
                         param_count=809_000_000, downloads=4_000_000, license="mit")]
    plan = mission.Planner(search=fake_search(pool), info=None, hardware=mac()).plan("Transcribe my meetings")
    assert plan.component("asr").candidate.model_id == "openai/whisper-large-v3-turbo"
    assert [s.key for s in plan.steps] == ["download"]
