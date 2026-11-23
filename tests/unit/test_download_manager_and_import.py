from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

import pytest

from noelle.app.services import download_manager as dm
from noelle.app.services.download_manager import DownloadJob, DownloadManager
from noelle.app.services.hf_client import ModelSummary
from noelle.app.services.library_import import (
    count_safetensors_params,
    discover_cached_models,
    import_cached_models,
    infer_pipeline_tag,
)
from noelle.app.services.model_registry import ModelRegistry
from noelle.config.settings import Settings


@pytest.fixture()
def manager(qapp: Any, tmp_path: Path) -> DownloadManager:
    return DownloadManager(Settings(tmp_path / "s.json"), ModelRegistry(tmp_path / "r.db"))


def _seed(manager: DownloadManager, repo: str, summary: ModelSummary | None = None) -> DownloadJob:
    job = DownloadJob(repo_id=repo, summary=summary)
    manager.jobs[repo] = job
    return job


def test_progress_and_registration(manager: DownloadManager, tmp_path: Path) -> None:
    job = _seed(manager, "org/m", ModelSummary(id="org/m", pipeline_tag="text-generation", library_name="transformers",
                                               license="mit", param_count=1000, tags=["license:mit"]))
    manager._apply("org/m", {"type": "plan", "commit": "c" * 40, "total_bytes": 1000, "download_bytes": 1000,
                             "files": [{"name": "model.safetensors", "size": 1000, "cached": False},
                                       {"name": "config.json", "size": 10, "cached": True}]})
    assert job.status == "downloading" and job.files["config.json"].cached
    manager._apply("org/m", {"type": "progress", "file": "model.safetensors", "downloaded": 500, "total": 1000})
    assert job.fraction == pytest.approx(0.5)
    manager._apply("org/m", {"type": "verify", "file": "model.safetensors", "ok": True})
    manager._apply("org/m", {"type": "file_done", "file": "model.safetensors", "size": 1000})
    assert job.files["model.safetensors"].verified is True and job.downloaded_bytes == 1000
    manager._apply("org/m", {"type": "retry", "file": "model.safetensors", "attempt": 1, "error": "timeout"})
    assert job.retries == 1
    manager._apply("org/m", {"type": "done", "path": str(tmp_path), "commit": "c" * 40, "total_bytes": 1010,
                             "files": ["model.safetensors", "config.json"]})
    assert job.status == "done" and job.fraction == 1.0
    record = manager.registry.get_model("org/m")
    assert record.pipeline_tag == "text-generation" and record.local_path == str(tmp_path)
    assert record.commit_hash == "c" * 40 and record.license == "mit" and record.size_bytes == 1010


def test_gguf_only_repo_registers_the_gguf_file(manager: DownloadManager, tmp_path: Path) -> None:
    _seed(manager, "org/g", ModelSummary(id="org/g", pipeline_tag="text-generation"))
    manager._apply("org/g", {"type": "done", "path": str(tmp_path), "total_bytes": 1,
                             "files": ["model-Q4_K_M.gguf", "README.md"]})
    record = manager.registry.get_model("org/g")
    assert record.gguf_file == "model-Q4_K_M.gguf" and record.precision == "gguf-q4_k_m"


def test_adapter_downloads_go_to_the_adapter_table(manager: DownloadManager, tmp_path: Path) -> None:
    _seed(manager, "org/lora", ModelSummary(id="org/lora", library_name="peft", base_models=["org/base"]))
    manager._apply("org/lora", {"type": "done", "path": str(tmp_path), "total_bytes": 5, "files": ["adapter.safetensors"]})
    assert manager.registry.get_model("org/lora") is None
    assert [a.adapter_id for a in manager.registry.list_adapters(base_model_id="org/base")] == ["org/lora"]


def test_error_event_sets_status(manager: DownloadManager) -> None:
    job = _seed(manager, "org/e")
    manager._apply("org/e", {"type": "error", "code": "DISK_FULL", "message": "Needs 9 GB"})
    assert job.status == "error" and job.error_code == "DISK_FULL" and not job.active


def test_launch_keeps_the_token_out_of_argv(manager: DownloadManager, monkeypatch: pytest.MonkeyPatch) -> None:
    started: dict[str, Any] = {}

    class FakeProcess:
        class ProcessState:
            NotRunning = 0

        def __init__(self, parent: Any = None) -> None:
            self.readyReadStandardOutput = self.readyReadStandardError = self.finished = _Sig()

        def setProcessEnvironment(self, env: Any) -> None:
            started["env"] = env

        def setProgram(self, program: str) -> None:
            started["program"] = program

        def setArguments(self, args: list[str]) -> None:
            started["args"] = args

        def start(self) -> None:
            started["started"] = True

        def state(self) -> int:
            return 1

    class _Sig:
        def connect(self, *_a: Any) -> None:
            pass

    monkeypatch.setattr(dm, "QProcess", FakeProcess)
    monkeypatch.setattr(dm.hf_client, "get_token", lambda: "hf_secret_token")
    manager.settings.set("storage.download_mode", "local_dir")
    manager.start("org/model", include=["*.safetensors"])
    args = started["args"]
    assert "download" in args and "org/model" in args and "--local-dir" in args and "--include" in args
    assert all("hf_secret_token" not in a for a in args)
    assert started["env"].value("HF_TOKEN") == "hf_secret_token"
    assert manager.job("org/model").status == "planning"


# --------------------------------------------------------------- library import
def _safetensors(path: Path, shapes: dict[str, list[int]]) -> None:
    header: dict[str, Any] = {}
    offset = 0
    for name, shape in shapes.items():
        size = 2
        for dim in shape:
            size *= dim
        header[name] = {"dtype": "F16", "shape": shape, "data_offsets": [offset, offset + size]}
        offset += size
    blob = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + b"\0" * offset)


def _repo(cache: Path, repo: str, files: dict[str, Any]) -> Path:
    snap = cache / f"models--{repo.replace('/', '--')}" / "snapshots" / ("a" * 40)
    snap.mkdir(parents=True)
    (snap.parent.parent / "refs").mkdir()
    (snap.parent.parent / "refs" / "main").write_text("a" * 40)
    for name, content in files.items():
        target = snap / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if name.endswith(".safetensors"):
            _safetensors(target, content)
        else:
            target.write_text(json.dumps(content) if isinstance(content, (dict, list)) else content)
    return snap


def test_import_cached_models_offline(tmp_path: Path) -> None:
    cache = tmp_path / "hub"
    _repo(cache, "org/llm", {"config.json": {"architectures": ["LlamaForCausalLM"]},
                             "model.safetensors": {"w": [4, 8], "b": [8]},
                             "README.md": "---\nlicense: apache-2.0\n---\n# hi"})
    _repo(cache, "org/embed", {"config.json": {"architectures": ["BertModel"]}, "modules.json": [],
                               "model.safetensors": {"w": [2, 2]}})
    _repo(cache, "org/sd", {"model_index.json": {"_class_name": "StableDiffusionPipeline"},
                            "unet/diffusion_pytorch_model.safetensors": {"w": [3]}})
    _repo(cache, "org/config-only", {"config.json": {"architectures": ["LlamaForCausalLM"]}})
    _repo(cache, "org/classifier", {"config.json": {"architectures": ["BertForSequenceClassification"]},
                                    "model.safetensors": {"w": [1]}})
    _repo(cache, "org/lora", {"adapter_config.json": {}, "config.json": {"architectures": ["LlamaForCausalLM"]},
                              "adapter_model.safetensors": {"w": [1]}})
    found = {r.repo_id: r for r in discover_cached_models(cache)}
    assert set(found) == {"org/llm", "org/embed", "org/sd"}
    assert found["org/llm"].pipeline_tag == "text-generation" and found["org/llm"].license == "apache-2.0"
    assert found["org/llm"].param_count == 40
    assert found["org/embed"].pipeline_tag == "sentence-similarity"
    assert found["org/sd"].pipeline_tag == "text-to-image" and found["org/sd"].library_name == "diffusers"
    registry = ModelRegistry(tmp_path / "r.db")
    assert sorted(import_cached_models(registry, cache)) == ["org/embed", "org/llm", "org/sd"]
    assert import_cached_models(registry, cache) == []  # idempotent


def test_task_inference_rules(tmp_path: Path) -> None:
    for arch, expected in (("WhisperForConditionalGeneration", "automatic-speech-recognition"),
                           ("VitsModel", "text-to-speech"), ("YolosForObjectDetection", "object-detection"),
                           ("ViTForImageClassification", "image-classification"),
                           ("Idefics3ForConditionalGeneration", "image-text-to-text")):
        snap = tmp_path / arch
        snap.mkdir()
        config = {"architectures": [arch]}
        if arch.startswith("Idefics3"):
            config["vision_config"] = {}
        (snap / "config.json").write_text(json.dumps(config))
        assert infer_pipeline_tag(snap) == expected, arch
    empty = tmp_path / "empty"
    empty.mkdir()
    assert count_safetensors_params(empty) is None
