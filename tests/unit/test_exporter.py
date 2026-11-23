"""Exporter tests.

Every task x precision variant must compile (and use no undefined names), user
strings must round-trip as literals, bundles must never contain a token, and the
generated server is exercised with fake runners. Integration tests run the
exported scripts against a model that is already in the Hugging Face cache
(read-only) and skip when it is absent.
"""

from __future__ import annotations

import ast
import base64
import builtins
import importlib
import importlib.metadata
import io
import json
import os
import re
import subprocess
import sys
import symtable
from pathlib import Path
from typing import Any, Iterator

import pytest

from noelle.app.services import exporter as ex

TASKS = ("text-generation", "image-text-to-text", "automatic-speech-recognition", "text-to-speech", "text-to-image",
         "sentence-similarity")
CASES = [(task, precision) for task in TASKS for precision in ex.supported_precisions(task)]
TRICKY_PROMPT = 'You are "Noelle".\nUse \'single\', """triple""", {braces}, a back\\slash, tab\t and \x00 nul.'
SENTINEL = "noelle-test-sentinel-not-a-real-token"
BUNDLE_FILES = ["inference.py", "requirements.txt", "Dockerfile", "server.py", "README.md"]
TEXT_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"  # same model as tests/conftest.py


@pytest.fixture(autouse=True)
def _guard_real_cache(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> None:
    """Generated scripts run in subprocesses must never write into the real Hugging Face cache."""
    guard = tmp_path_factory.mktemp("hf-guard")
    monkeypatch.setenv("HF_HOME", str(guard / "home"))
    monkeypatch.setenv("HF_HUB_CACHE", str(guard / "hub"))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")


def make_config(task: str = "text-generation", precision: str = "auto", **overrides: Any) -> ex.ExportConfig:
    values: dict[str, Any] = {
        "model_id": "org/model-1.5B",
        "task": task,
        "precision": precision,
        "system_prompt": TRICKY_PROMPT,
        "params": {"temperature": 0.0, "top_p": 0.8, "top_k": 20, "repetition_penalty": 1.05, "max_new_tokens": 64,
                   "seed": 7},
        "local_path": "/nonexistent/noelle/snapshot",
        "gguf_file": "model-Q4_K_M.gguf" if precision == "gguf-q4_k_m" else None,
        "adapter_id": "org/lora-adapter",
        "extra": {"steps": 20, "guidance": 5.5, "width": 768, "height": 512, "negative_prompt": 'no "blur"',
                  "seed": 42, "language": "en", "speed": 1.25},
    }
    values.update(overrides)
    return ex.ExportConfig(**values)


def undefined_names(source: str, filename: str) -> set[str]:
    """Global names that are read somewhere but never bound at module level (mini pyflakes)."""
    top = symtable.symtable(source, filename, "exec")
    bound = {s.get_name() for s in top.get_symbols() if s.is_assigned() or s.is_imported()}
    allowed = bound | set(dir(builtins)) | {"__file__", "__name__"}
    missing: set[str] = set()

    def visit(table: symtable.SymbolTable) -> None:
        module_level = table.get_type() == symtable.SymbolTableType.MODULE if hasattr(symtable, "SymbolTableType") \
            else table.get_type() == "module"
        for symbol in table.get_symbols():
            if symbol.is_referenced() and (module_level or symbol.is_global()) and symbol.get_name() not in allowed:
                missing.add(symbol.get_name())
        for child in table.get_children():
            visit(child)

    visit(top)
    return missing


def assignments(source: str) -> dict[str, Any]:
    """Module-level ``NAME = <literal>`` assignments of a generated script."""
    values: dict[str, Any] = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                values[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                continue
    return values


def heredoc(dockerfile: str) -> str:
    match = re.search(r"<<'PY'\n(.*?)\nPY\n", dockerfile, re.S)
    assert match, "build-time download script not found"
    return match.group(1)


# ---------------------------------------------------------------------------
# Generated code
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("task,precision", CASES)
def test_every_variant_compiles_without_undefined_names(task: str, precision: str) -> None:
    config = make_config(task, precision)
    for name, source in (("inference.py", ex.python_script(config)), ("server.py", ex.server_script(config))):
        compile(source, name, "exec")
        ast.parse(source, feature_version=(3, 12))  # the container runs Python 3.12
        assert undefined_names(source, name) == set(), name
    compile(heredoc(ex.dockerfile(config)), "Dockerfile:download", "exec")


@pytest.mark.parametrize("task,precision", CASES)
def test_user_values_round_trip_as_literals(task: str, precision: str) -> None:
    config = make_config(task, precision)
    values = assignments(ex.python_script(config))
    assert values["MODEL_ID"] == "org/model-1.5B"
    assert values["SYSTEM_PROMPT"] == TRICKY_PROMPT
    assert values["LOCAL_PATH"] == "/nonexistent/noelle/snapshot"
    assert values["TASK"] == task
    assert values["EXTRA"]["negative_prompt"] == 'no "blur"'
    assert values["GENERATION"] == {"temperature": 0.0, "top_p": 0.8, "top_k": 20, "repetition_penalty": 1.05,
                                    "max_new_tokens": 64, "seed": 7}
    assert values["PRECISION"] in ex.supported_precisions(task)


def test_backend_selection() -> None:
    def script(**kwargs: Any) -> str:
        return ex.python_script(make_config(**kwargs))

    assert "AutoModelForCausalLM" in script(precision="fp16") and "TextIteratorStreamer" in script()
    assert "BitsAndBytesConfig(load_in_4bit=True" in script(precision="nf4")
    mlx = script(precision="mlx-4bit")
    assert "from mlx_lm import load" in mlx and "make_sampler" in mlx and "BASE_MODEL_ID" in mlx
    gguf = script(precision="gguf-q4_k_m")
    assert "from llama_cpp import Llama" in gguf and assignments(gguf)["GGUF_FILE"] == "model-Q4_K_M.gguf"
    assert "import torch" not in gguf
    assert "AutoModelForImageTextToText" in script(task="image-text-to-text")
    asr = script(task="automatic-speech-recognition")
    assert '"automatic-speech-recognition"' in asr and "soundfile" in asr and "ffmpeg" not in asr.replace(
        "no ffmpeg", "")
    assert 'pipeline("text-to-audio"' in script(task="text-to-speech")
    image = script(task="text-to-image")
    assert "AutoPipelineForText2Image" in image and "manual_seed(seed)" in image and ".save(" in image
    assert "SentenceTransformer" in script(task="sentence-similarity")
    assert "torch_dtype" not in "".join(script(task=t) for t in TASKS)  # transformers 5 / diffusers: dtype=


def test_irrelevant_precision_falls_back_with_a_note() -> None:
    source = ex.python_script(make_config("text-to-image", "gguf-q4_k_m"))
    assert assignments(source)["PRECISION"] == "auto"
    assert "# Note: GGUF Q4_K_M is not available for text-to-image" in source
    hostile = ex.python_script(make_config(precision="banana\nimport shutil"))
    assert [line for line in hostile.splitlines() if "banana" in line] == \
        ["# Note: Unknown precision 'banana\\nimport shutil'; exported with auto."]  # stays inside one comment
    assert "import shutil" not in [line.strip() for line in hostile.splitlines()]
    compile(hostile, "inference.py", "exec")


def test_invalid_configurations_are_rejected() -> None:
    with pytest.raises(ValueError):
        ex.python_script(make_config("object-detection"))
    with pytest.raises(ValueError):
        ex.dockerfile(make_config(model_id="org/model\nRUN rm -rf /"))
    with pytest.raises(ValueError):
        ex.export_bundle(make_config(model_id=""), Path("/nonexistent"))
    with pytest.raises(ValueError):
        ex.supported_precisions("object-detection")


def exec_script(source: str) -> dict[str, Any]:
    namespace: dict[str, Any] = {"__name__": "exported", "__file__": "/tmp/exported/inference.py"}
    exec(compile(source, "inference.py", "exec"), namespace)  # defines functions only: heavy imports are lazy
    return namespace


def test_sampling_matches_the_playground() -> None:
    ns = exec_script(ex.python_script(make_config()))
    greedy = ns["generation_kwargs"](ns["sampling"]())
    assert greedy == {"max_new_tokens": 64, "repetition_penalty": 1.05, "do_sample": False, "temperature": None,
                      "top_p": None, "top_k": None}
    sampled = ns["generation_kwargs"](ns["sampling"]({"temperature": 0.7, "top_k": 0}))
    assert (sampled["do_sample"], sampled["temperature"], sampled["top_p"], sampled["top_k"]) == (True, 0.7, 0.8, None)
    clamped = ns["sampling"]({"temperature": 9, "top_p": 0, "repetition_penalty": 5, "max_new_tokens": 0})
    assert (clamped["temperature"], clamped["top_p"], clamped["repetition_penalty"], clamped["max_new_tokens"]) == \
        (2.0, 0.01, 2.5, 1)
    assert ns["sampling"]()["seed"] == 7
    merged = ns["merge_system"]([{"role": "system", "content": "S"}, {"role": "user", "content": "U"}])
    assert merged == [{"role": "user", "content": "S\n\nU"}]
    assert ns["plain_prompt"]([{"role": "user", "content": "Hi"}]) == "User: Hi\n\nAssistant:"
    assert ns["model_source"]() == "org/model-1.5B"  # LOCAL_PATH missing on this machine
    assert ns["model_source"](None, str(Path.home())) == str(Path.home())


def test_playground_defaults_apply_when_params_are_missing() -> None:
    values = assignments(ex.python_script(make_config(params={})))
    assert values["GENERATION"] == {"temperature": 0.7, "top_p": 0.9, "top_k": 50, "repetition_penalty": 1.1,
                                    "max_new_tokens": 512, "seed": None}


def test_tts_and_asr_helpers_run_without_models() -> None:
    np = pytest.importorskip("numpy")
    tts = exec_script(ex.python_script(make_config("text-to-speech")))
    tone = np.sin(np.linspace(0, 200 * np.pi, 16000)).astype(np.float32)
    assert abs(len(tts["time_stretch"](tone, 1.25)) - 16000 / 1.25) < 1100
    assert float(np.max(np.abs(tts["peak_normalize"](tone * 0.1)))) == pytest.approx(0.95, rel=1e-4)
    sf = pytest.importorskip("soundfile")
    asr = exec_script(ex.python_script(make_config("automatic-speech-recognition")))
    buffer = io.BytesIO()
    sf.write(buffer, np.stack([tone, tone], axis=1)[:8000], 8000, format="WAV")  # stereo, 8 kHz
    buffer.seek(0)
    audio = asr["load_audio"](buffer, 16000)
    assert audio.dtype == np.float32 and audio.ndim == 1 and abs(audio.shape[0] - 16000) <= 1


# ---------------------------------------------------------------------------
# requirements.txt and headers
# ---------------------------------------------------------------------------


def test_requirements_are_pinned_from_the_environment() -> None:
    text = ex.requirements_txt(make_config(precision="nf4"))
    lines = [line for line in text.splitlines() if line and not line.startswith("#")]
    for line in lines:
        name = line.split("==")[0].split(";")[0].strip()
        if ex.installed_version(name):
            assert line.startswith(f"{name}=={ex.installed_version(name)}"), line
        else:
            assert line == name, line  # unpinned fallback for packages missing here
    names = {line.split("==")[0].split(" ;")[0] for line in lines}
    assert {"torch", "transformers", "accelerate", "bitsandbytes", "peft", "fastapi", "uvicorn"} <= names
    assert f"transformers=={importlib.metadata.version('transformers')}" in lines


def test_pinning_strips_local_versions_and_falls_back_unpinned(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = {"torch": "2.5.1+cu124", "transformers": "5.24.0"}

    def version(name: str) -> str:
        if name in fake:
            return fake[name]
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(ex.importlib.metadata, "version", version)
    lines = ex.requirements_txt(make_config(precision="fp16")).splitlines()
    assert "torch==2.5.1" in lines and "transformers==5.24.0" in lines and "accelerate" in lines
    header = ex.python_script(make_config(precision="fp16"))
    assert 'pip install "torch==2.5.1" "transformers==5.24.0" "accelerate"' in header


def test_requirements_per_backend() -> None:
    def names(**kwargs: Any) -> set[str]:
        lines = ex.requirements_txt(make_config(**kwargs)).splitlines()
        return {line.split("==")[0].split(" ;")[0] for line in lines if line and not line.startswith("#")}

    assert {"llama-cpp-python", "huggingface-hub", "fastapi"} <= names(precision="gguf-q4_k_m")
    assert "torch" not in names(precision="gguf-q4_k_m")
    assert {"gptqmodel", "optimum"} <= names(precision="gptq-4bit")
    assert "gptqmodel" in names(precision="awq-4bit")
    assert {"diffusers", "pillow", "peft"} <= names(task="text-to-image")
    assert {"soundfile", "scipy", "numpy"} <= names(task="automatic-speech-recognition")
    assert "peft" not in names(task="automatic-speech-recognition")
    assert "sentence-transformers" in names(task="sentence-similarity")
    mlx = ex.requirements_txt(make_config(precision="mlx-4bit"))
    assert re.search(r'^mlx-lm(==\S+)? ; sys_platform == "darwin" and platform_machine == "arm64"$', mlx, re.M)
    assert re.search(r'^torch(==\S+)? ; sys_platform != "darwin" or platform_machine != "arm64"$', mlx, re.M)


def test_header_has_pinned_install_line() -> None:
    source = ex.python_script(make_config(precision="bf16"))
    torch_version = ex.installed_version("torch")
    expected = f'"torch=={torch_version}"' if torch_version else '"torch"'
    assert re.search(r"^#   pip install .*" + re.escape(expected), source, re.M)
    assert "fastapi" not in source.split("from __future__")[0]  # server packages stay in requirements.txt


# ---------------------------------------------------------------------------
# Dockerfile and bundle
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("task,precision", [("text-generation", "auto"), ("text-generation", "mlx-4bit"),
                                            ("text-generation", "gguf-q4_k_m"), ("text-generation", "nf4"),
                                            ("image-text-to-text", "fp16"), ("automatic-speech-recognition", "auto"),
                                            ("text-to-speech", "auto"), ("text-to-image", "fp16"),
                                            ("sentence-similarity", "auto")])
def test_bundle_is_written_compiles_and_never_contains_a_token(task: str, precision: str, tmp_path: Path,
                                                               monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_TOKEN", SENTINEL)
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", SENTINEL)
    out = tmp_path / "bundle"
    written = ex.export_bundle(make_config(task, precision), out)
    assert written == [out / name for name in BUNDLE_FILES]
    assert all(path.is_file() and path.stat().st_size > 0 for path in written)
    assert os.access(out / "inference.py", os.X_OK)
    for path in written:
        assert SENTINEL not in path.read_text(encoding="utf-8"), path.name
    check = subprocess.run([sys.executable, "-m", "py_compile", "inference.py", "server.py"], cwd=out,
                           capture_output=True, text=True)
    assert check.returncode == 0, check.stderr

    docker = (out / "Dockerfile").read_text()
    assert docker.startswith("# syntax=docker/dockerfile:1\n")
    assert re.search(r"^FROM nvidia/cuda:[\w.-]+-runtime-ubuntu24\.04$", docker, re.M)
    assert re.search(r"^RUN --mount=type=secret,id=hf_token\b", docker, re.M)
    assert "/run/secrets/hf_token" in docker
    assert not re.search(r"^\s*(ARG|ENV)\b[^\n]*HF_TOKEN", docker, re.M)  # never baked into a layer
    assert "pip install -r requirements.txt" in docker and "python3 -m venv /opt/venv" in docker
    assert re.search(r"^USER noelle$", docker, re.M) and re.search(r"^EXPOSE 8000$", docker, re.M)
    assert re.search(r"^HEALTHCHECK .*\n\s+CMD .*/health", docker, re.M)
    assert 'ENTRYPOINT ["uvicorn", "server:app"]' in docker and '"--host", "0.0.0.0"' in docker
    assert docker.index("--mount=type=secret") < docker.index("ENV HF_HUB_OFFLINE=1") < docker.index("COPY inference.py")
    assert docker.index("USER noelle") < docker.index("--mount=type=secret")
    compile(heredoc(docker), "download", "exec")
    readme = (out / "README.md").read_text()
    assert "pip install -r requirements.txt" in readme and "--secret id=hf_token" in readme


def test_mlx_docker_image_falls_back_to_the_base_model() -> None:
    config = make_config(model_id="mlx-community/Qwen2.5-0.5B-Instruct-4bit", precision="mlx-4bit",
                         extra={"base_model": "Qwen/Qwen2.5-0.5B-Instruct"})
    docker = ex.dockerfile(config)
    assert "Containers cannot use Metal" in docker
    assert "repo_id = 'Qwen/Qwen2.5-0.5B-Instruct'" in heredoc(docker)
    source = ex.python_script(config)
    assert assignments(source)["BASE_MODEL_ID"] == "Qwen/Qwen2.5-0.5B-Instruct"
    assert 'precision="fp16"' in source and "is_apple_silicon()" in source
    unknown = ex.python_script(make_config(model_id="mlx-community/Some-Model-4bit", precision="mlx-4bit", extra={}))
    assert "# Note: MODEL_ID is an MLX conversion" in unknown


def test_gguf_download_fetches_a_single_file() -> None:
    script = heredoc(ex.dockerfile(make_config(precision="gguf-q4_k_m")))
    assert "hf_hub_download(repo_id, filename, token=token)" in script and "snapshot_download(repo_id" not in script
    assert "build-essential cmake" in ex.dockerfile(make_config(precision="gguf-q4_k_m"))


# ---------------------------------------------------------------------------
# server.py with fake runners (no model needed)
# ---------------------------------------------------------------------------


@pytest.fixture
def load_server(tmp_path: Path) -> Iterator[Any]:
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    added: list[str] = []

    def load(config: ex.ExportConfig, runner: Any) -> Any:
        from fastapi.testclient import TestClient

        out = tmp_path / f"bundle-{len(added)}"
        ex.export_bundle(config, out)
        for name in ("server", "inference"):
            sys.modules.pop(name, None)
        sys.path.insert(0, str(out))
        added.append(str(out))
        server = importlib.import_module("server")
        client = TestClient(server.app)  # no lifespan: the fake runner stands in for the model
        assert client.get("/health").status_code == 503
        if runner is not None:
            server.STATE["runner"] = runner
            health = client.get("/health")
            assert health.status_code == 200 and health.json()["model"] == config.model_id
        return client

    yield load
    for path in added:
        sys.path.remove(path)
    for name in ("server", "inference"):
        sys.modules.pop(name, None)


class FakeChat:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []

    def stream(self, *args: Any, **overrides: Any) -> Iterator[str]:
        self.calls.append((args, overrides))
        yield from ("Hel", "lo")


def test_server_chat_completions(load_server: Any) -> None:
    runner = FakeChat()
    client = load_server(make_config(system_prompt="Be brief."), runner)
    body = {"messages": [{"role": "user", "content": [{"type": "text", "text": "Hi"}]}], "max_tokens": 5,
            "temperature": 0}
    reply = client.post("/v1/chat/completions", json=body).json()
    assert reply["object"] == "chat.completion" and reply["choices"][0]["message"]["content"] == "Hello"
    (messages,), overrides = runner.calls[-1]
    assert messages == [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"}]
    assert overrides["max_new_tokens"] == 5 and overrides["temperature"] == 0
    stream = client.post("/v1/chat/completions", json={**body, "stream": True})
    events = [line[6:] for line in stream.text.splitlines() if line.startswith("data: ")]
    assert events[-1] == "[DONE]"
    assert "".join(json.loads(e)["choices"][0]["delta"].get("content", "") for e in events[:-1]) == "Hello"
    assert client.post("/v1/chat/completions", json={"messages": []}).status_code == 400


def test_server_vision_chat(load_server: Any) -> None:
    image_mod = pytest.importorskip("PIL.Image")
    runner = FakeChat()
    client = load_server(make_config("image-text-to-text"), runner)
    buffer = io.BytesIO()
    image_mod.new("RGB", (4, 3), "red").save(buffer, format="PNG")
    url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
    content = [{"type": "text", "text": "What color?"}, {"type": "image_url", "image_url": {"url": url}}]
    reply = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": content}]})
    assert reply.status_code == 200 and reply.json()["choices"][0]["message"]["content"] == "Hello"
    (image, prompt, system), _ = runner.calls[-1]
    assert image.size == (4, 3) and prompt == "What color?" and system == TRICKY_PROMPT
    missing = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})
    assert missing.status_code == 400
    bad = [{"type": "image_url", "image_url": {"url": "file:///etc/passwd"}}]
    assert client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": bad}]}).status_code == 400


def test_server_transcriptions(load_server: Any) -> None:
    np = pytest.importorskip("numpy")
    sf = pytest.importorskip("soundfile")

    class FakeAsr:
        sampling_rate = 16000

        def transcribe(self, audio: Any, **kwargs: Any) -> dict:
            return {"text": f"{audio.shape[0]} samples", "chunks": [], "language": kwargs["language"]}

    client = load_server(make_config("automatic-speech-recognition"), FakeAsr())
    buffer = io.BytesIO()
    sf.write(buffer, np.zeros(8000, dtype=np.float32), 8000, format="WAV")
    reply = client.post("/v1/audio/transcriptions", content=buffer.getvalue())
    assert reply.status_code == 200 and reply.json()["text"] == "16000 samples" and reply.json()["language"] == "en"
    assert client.post("/v1/audio/transcriptions", content=b"not audio").status_code == 400
    assert client.post("/v1/audio/transcriptions", content=b"").status_code == 400


def test_server_speech(load_server: Any) -> None:
    np = pytest.importorskip("numpy")

    class FakeTts:
        def synthesize(self, text: str, **kwargs: Any) -> tuple[Any, int]:
            return np.zeros(1600, dtype=np.float32), 16000

    client = load_server(make_config("text-to-speech"), FakeTts())
    reply = client.post("/v1/audio/speech", json={"input": "Hello", "speed": 1.2})
    assert reply.status_code == 200 and reply.headers["content-type"] == "audio/wav"
    assert reply.content[:4] == b"RIFF"
    assert client.post("/v1/audio/speech", json={"input": "  "}).status_code == 400


def test_server_images(load_server: Any) -> None:
    image_mod = pytest.importorskip("PIL.Image")

    class FakeDiffusion:
        def generate(self, prompt: str, **kwargs: Any) -> tuple[Any, int]:
            return image_mod.new("RGB", (kwargs["width"], kwargs["height"])), 42

    client = load_server(make_config("text-to-image"), FakeDiffusion())
    reply = client.post("/v1/images/generations", json={"prompt": "a cat", "size": "64x32"}).json()
    png = base64.b64decode(reply["data"][0]["b64_json"])
    assert png[:8] == b"\x89PNG\r\n\x1a\n" and reply["data"][0]["seed"] == 42
    assert image_mod.open(io.BytesIO(png)).size == (64, 32)
    assert client.post("/v1/images/generations", json={"prompt": "x", "size": "big"}).status_code == 400


def test_server_embeddings(load_server: Any) -> None:
    np = pytest.importorskip("numpy")

    class FakeEmbedder:
        def embed(self, texts: list[str]) -> Any:
            return np.ones((len(texts), 3), dtype=np.float32)

        def similarity(self, query: str, sentences: list[str]) -> list[float]:
            return [0.5 for _ in sentences]

    client = load_server(make_config("sentence-similarity"), FakeEmbedder())
    reply = client.post("/v1/embeddings", json={"input": ["a", "b"]}).json()
    assert [item["index"] for item in reply["data"]] == [0, 1] and reply["data"][0]["embedding"] == [1.0, 1.0, 1.0]
    assert client.post("/v1/similarity", json={"source": "q", "sentences": ["x", "y"]}).json()["scores"] == [0.5, 0.5]


# ---------------------------------------------------------------------------
# Integration: run the exports against a cached model (read-only)
# ---------------------------------------------------------------------------


def run_inference(out: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    return subprocess.run([sys.executable, "inference.py", *args], cwd=out, env=env, capture_output=True, text=True,
                          stdin=subprocess.DEVNULL, timeout=600)


@pytest.mark.integration
def test_exported_transformers_script_generates_text(text_model_path: Path, tmp_path: Path) -> None:
    out = tmp_path / "bundle"
    ex.export_bundle(ex.ExportConfig(TEXT_MODEL, "text-generation", "fp32", device="cpu",
                                     system_prompt="Answer in one short sentence.",
                                     params={"temperature": 0, "max_new_tokens": 8},
                                     local_path=str(text_model_path)), out)
    result = run_inference(out, "What is the capital of France?")
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.strip()


@pytest.mark.integration
@pytest.mark.skipif(sys.platform != "darwin" or os.uname().machine != "arm64", reason="MLX needs Apple Silicon")
def test_exported_mlx_script_generates_text(text_model_path: Path, tmp_path: Path) -> None:
    pytest.importorskip("mlx_lm")
    out = tmp_path / "bundle"
    ex.export_bundle(ex.ExportConfig(TEXT_MODEL, "text-generation", "mlx-4bit",
                                     params={"temperature": 0, "max_new_tokens": 8},
                                     local_path=str(text_model_path)), out)
    result = run_inference(out, "Say hello.")
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.strip()
    assert not (out / "mlx-4bit").exists()  # the app's folder was used, nothing converted


@pytest.mark.integration
def test_exported_server_serves_a_real_model(text_model_path: Path, load_server: Any, tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    config = ex.ExportConfig(TEXT_MODEL, "text-generation", "fp32", device="cpu",
                             params={"temperature": 0, "max_new_tokens": 6}, local_path=str(text_model_path))
    client = load_server(config, None)
    server = sys.modules["server"]
    server.STATE["runner"] = None
    with TestClient(server.app) as live:  # lifespan loads the model through inference.load_runner
        assert live.get("/health").json()["status"] == "ok"
        reply = live.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "Hi!"}]})
        assert reply.status_code == 200 and reply.json()["choices"][0]["message"]["content"].strip()
    del client


def test_exported_vision_server_refuses_remote_and_private_image_urls(monkeypatch) -> None:
    """SSRF guard in the generated server: remote images are off unless allow-listed, never private."""
    import pytest

    from noelle.app.services.exporter import ExportConfig, server_script

    code = server_script(ExportConfig(model_id="org/vlm", task="image-text-to-text"))
    start = code.index("MAX_IMAGE_BYTES = ")
    end = code.index("def first_image(content):")
    snippet = code[start:end]

    class HTTPException(Exception):
        def __init__(self, status_code: int, detail: str) -> None:
            super().__init__(detail)
            self.status_code = status_code

    import base64
    import io
    import urllib.request

    namespace = {"HTTPException": HTTPException, "base64": base64, "io": io, "urllib": urllib}
    monkeypatch.delenv("ALLOW_REMOTE_IMAGE_URLS", raising=False)
    exec(compile(snippet, "server_snippet", "exec"), namespace)
    load = namespace["load_image_url"]
    for url in ("https://example.com/cat.png", "http://169.254.169.254/latest/meta-data/", "file:///etc/passwd"):
        with pytest.raises(HTTPException) as err:
            load(url)
        assert err.value.status_code == 400
    namespace["ALLOW_REMOTE_IMAGES"] = True
    namespace["REMOTE_IMAGE_HOSTS"] = {"localhost"}
    with pytest.raises(HTTPException) as err:
        load("https://localhost/cat.png")  # allow-listed but resolves to loopback
    assert "non-public" in str(err.value)
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (4, 4), "red").save(buf, format="PNG")
    image = load("data:image/png;base64," + base64.b64encode(buf.getvalue()).decode())
    assert image.size == (4, 4)
