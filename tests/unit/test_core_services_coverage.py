"""Behavioural tests for the remaining paths of the four core services named in the Definition of Done."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from noelle.app.services import hardware_manager as hm
from noelle.app.services import hf_client
from noelle.app.services.download_manager import DownloadJob, DownloadManager, FileProgress
from noelle.app.services.hf_client import HFClient, ModelSummary
from noelle.app.services.model_registry import ModelRegistry
from noelle.config.settings import Settings
from noelle.runtime.preflight import GB


# ===================================================================== hardware
class _Runtime:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.last_heartbeat_payload = payload
        self.state = "ready"
        self.heartbeat = SimpleNamespace(connect=lambda fn: setattr(self, "_beat", fn))


def test_memory_budget_prefers_worker_report(qapp: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hm, "detect_gpu", lambda: hm.GpuInfo("mps", "Apple GPU", int(12 * GB)))
    runtime = _Runtime({"budget": {"total": int(10 * GB)}, "device": "mps"})
    manager = hm.HardwareManager(runtime)
    budget = manager.memory_budget(0.2)
    assert budget.total_bytes == int(10 * GB) and budget.usable_bytes == int(8 * GB) and budget.device == "mps"
    runtime.last_heartbeat_payload = {}
    assert manager.memory_budget().total_bytes == int(12 * GB)
    manager.gpu = hm.GpuInfo("cpu", "CPU", None)
    assert manager.memory_budget().device == "cpu"


def test_publish_merges_worker_memory(qapp: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hm, "detect_gpu", lambda: hm.GpuInfo("mps", "Apple GPU", int(12 * GB)))
    runtime = _Runtime({"memory": {"total": int(11 * GB), "allocated": 100, "peak": 200},
                        "budget": {"usable": int(9 * GB)}, "loaded": [{"model_id": "m"}], "device": "mps"})
    manager = hm.HardwareManager(runtime)
    seen: list[Any] = []
    manager.hardware_telemetry_updated.connect(seen.append)
    sample = manager._sample(True)
    assert {"cpu_percent", "ram_total", "ram_used", "ram_available", "gpu_sample"} <= set(sample)
    manager._publish({"cpu_percent": 12.5, "ram_total": 16 * GB, "ram_used": 8 * GB, "ram_available": 8 * GB,
                      "gpu_sample": (33.0, 5 * GB)})
    metrics = seen[-1]
    assert metrics.gpu_util == 33.0 and metrics.vram_used == 5 * GB and metrics.vram_total == int(11 * GB)
    assert metrics.worker_allocated == 100 and metrics.worker_peak == 200 and metrics.budget_usable == int(9 * GB)
    assert metrics.loaded_models == [{"model_id": "m"}] and metrics.runtime_state == "ready"
    runtime._beat({"device": "mps", "memory": {"total": int(13 * GB)}})
    assert manager.gpu.total_memory == int(13 * GB)
    assert hm.describe_gb(None) == "—" and hm.describe_gb(int(2 * GB)) == "2.0 GB"


def test_detect_gpu_fallbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hm.shutil, "which", lambda _n: None)
    monkeypatch.setattr(hm.platform, "system", lambda: "Linux")
    assert hm.detect_gpu().kind == "cpu"
    monkeypatch.setattr(hm.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(hm.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(hm, "_run", lambda args, timeout=2.0: '"model" = "Apple M9"\n"gpu-core-count" = 40')
    gpu = hm.detect_gpu()
    assert gpu.kind == "mps" and gpu.name == "Apple M9 GPU" and gpu.cores == 40
    assert hm.sample_gpu(hm.GpuInfo("cpu", "x")) == (None, None)
    monkeypatch.setattr(hm, "_run", lambda args, timeout=2.0: "garbage")
    assert hm.sample_gpu(hm.GpuInfo("cuda", "x")) == (None, None)


def test_poll_runs_off_the_gui_thread(qtbot: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hm, "detect_gpu", lambda: hm.GpuInfo("cpu", "CPU", None))
    manager = hm.HardwareManager(None, interval_ms=50)
    with qtbot.waitSignal(manager.hardware_telemetry_updated, timeout=3000):
        manager.start()
    manager.stop()
    assert manager.latest is not None and manager.latest.runtime_state == "stopped"


def test_hardware_manager_run_no_window_on_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []

    def fake_run(args: list[str], **kwargs: Any) -> Any:
        calls.append(kwargs)
        from types import SimpleNamespace
        return SimpleNamespace(stdout="ok")

    monkeypatch.setattr(hm.sys, "platform", "win32")
    monkeypatch.setattr(hm.subprocess, "run", fake_run)
    assert hm._run(["nvidia-smi"]) == "ok"
    assert len(calls) == 1
    assert calls[0].get("creationflags") == getattr(hm.subprocess, "CREATE_NO_WINDOW", 0x08000000)


def test_worker_command_prefers_pythonw_on_windows(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    from noelle.runtime.client import worker_command

    scripts = tmp_path / "Scripts"
    scripts.mkdir()
    python_exe = scripts / "python.exe"
    python_exe.touch()
    pythonw_exe = scripts / "pythonw.exe"
    pythonw_exe.touch()

    monkeypatch.setattr(sys, "frozen", False, raising=False)
    monkeypatch.setattr("sys.platform", "win32")
    cmd = worker_command(str(python_exe))
    assert cmd == [str(pythonw_exe), "-m", "noelle"]


def test_main_does_not_attach_console_for_gui(monkeypatch: pytest.MonkeyPatch) -> None:
    from noelle import __main__ as nmain
    import noelle.app.main as app_main

    attached = []
    monkeypatch.setattr(nmain, "_attach_windows_console", lambda: attached.append(True))
    monkeypatch.setattr(app_main, "run", lambda argv: 0)

    for cmd in (["ui"], ["setup"]):
        attached.clear()
        nmain.main(cmd)
        assert attached == []


# ==================================================================== hf_client
def _info(**overrides: Any) -> Any:
    base = dict(id="org/Model-2B", author=None, pipeline_tag="text-generation", library_name="peft",
                downloads=5, likes=1, trending_score=3, tags=["base_model:adapter:org/base", "base_model:org/base"],
                safetensors=None, gguf={"total": 2_000_000_000, "context_length": 4096}, gated="auto",
                created_at=datetime(2025, 1, 1, tzinfo=timezone.utc), last_modified=None,
                config={"model_type": "llama", "quantization_config": {"bits": 4}},
                card_data=SimpleNamespace(to_dict=lambda: {"license": ["mit", "apache-2.0"]}),
                siblings=[SimpleNamespace(rfilename="w.gguf", size=None, lfs=SimpleNamespace(size=123))])
    base.update(overrides)
    return SimpleNamespace(**base)


def test_summary_properties() -> None:
    s = ModelSummary.from_info(_info())
    assert s.namespace == "org" and s.name == "Model-2B"
    assert s.license == "mit, apache-2.0" and s.param_count == 2_000_000_000
    assert s.context_length == 4096 and s.is_adapter and s.prequantized and s.architecture == "llama"
    assert s.base_models == ["org/base", "org/base"] and s.siblings == [{"name": "w.gguf", "size": 123}]
    assert s.gated == "auto"


def test_sort_keys_and_multi_tag_merge(monkeypatch: pytest.MonkeyPatch) -> None:
    older = ModelSummary(id="a/old", downloads=1, likes=9, trending_score=1,
                         last_modified=datetime(2024, 1, 1, tzinfo=timezone.utc))
    newer = ModelSummary(id="b/new", downloads=9, likes=1, trending_score=5,
                         last_modified=datetime(2026, 1, 1, tzinfo=timezone.utc))
    assert hf_client._sort_key("downloads")(newer) > hf_client._sort_key("downloads")(older)
    assert hf_client._sort_key("likes")(older) > hf_client._sort_key("likes")(newer)
    assert hf_client._sort_key("lastModified")(newer) > hf_client._sort_key("lastModified")(older)
    assert hf_client._sort_key("trendingScore")(newer) == 5

    class Response:
        def __init__(self, items: list[dict[str, Any]]) -> None:
            self._items, self.links, self.status_code = items, {}, 200

        def json(self) -> list[dict[str, Any]]:
            return self._items

    class Session:
        def get(self, url: str, params: Any = None, headers: Any = None, timeout: Any = None) -> Response:
            tag = params["pipeline_tag"]
            assert params["limit"] == 10  # page size split across the merged tags
            return Response([{"id": f"x/{tag}", "pipeline_tag": tag, "downloads": len(tag)}])

    import huggingface_hub.utils as hub_utils

    monkeypatch.setattr(hub_utils, "get_session", lambda: Session())
    monkeypatch.setattr(hub_utils, "hf_raise_for_status", lambda r: None)
    pager = HFClient().pager(modality="audio", sort="downloads", page_size=30)
    models = pager.fetch_next()
    assert [m.pipeline_tag for m in models] == ["automatic-speech-recognition", "text-to-speech", "text-to-audio"]
    assert pager.exhausted and pager.fetch_next() == []


def test_client_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    class Api:
        def __init__(self, endpoint: Any = None, token: Any = None) -> None:
            pass

        def model_info(self, repo_id: str, files_metadata: bool = False) -> Any:
            assert files_metadata
            return _info(id=repo_id)

        def whoami(self, token: Any = None) -> dict[str, Any]:
            return {"name": "tester"}

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "HfApi", Api)
    client = HFClient()
    assert client.model_info("org/x").id == "org/x"
    assert client.whoami()["name"] == "tester"

    class Resp:
        def __init__(self, status: int, content: bytes) -> None:
            self.status_code, self.content = status, content
            self.read_chunks = 0

        def __enter__(self) -> Resp:
            return self

        def __exit__(self, *_exc: Any) -> None:
            return None

        def iter_bytes(self) -> Any:
            for start in range(0, len(self.content), 4):
                self.read_chunks += 1
                yield self.content[start:start + 4]

    huge = Resp(200, b"x" * 10_000)
    responses = {"config.json": Resp(200, b'{"max_position_embeddings": 4096}'), "README.md": Resp(200, b"# Card"),
                 "bad.json": Resp(200, b"{not json"), "HUGE.md": huge}

    class Session:
        def stream(self, method: str, url: str, **kwargs: Any) -> Resp:
            assert method == "GET" and kwargs["follow_redirects"] is True
            return responses.get(url.rsplit("/", 1)[-1], Resp(404, b""))

    import huggingface_hub.utils as hub_utils

    monkeypatch.setattr(hub_utils, "get_session", lambda: Session())
    assert client.fetch_config("org/x") == {"max_position_embeddings": 4096}
    assert client.fetch_readme("org/x") == "# Card"
    assert client.fetch_file_text("org/x", "missing.txt") is None
    assert client.fetch_file_text("org/x", "HUGE.md", max_bytes=10) == "x" * 10
    assert huge.read_chunks == 3  # stopped streaming instead of reading all 10 000 bytes
    monkeypatch.setitem(responses, "config.json", Resp(200, b"{not json"))
    assert client.fetch_config("org/x") == {}

    monkeypatch.setattr(hf_client.ModelPager, "fetch_next", lambda self: [
        ModelSummary(id="a/b", pipeline_tag="text-generation"), ModelSummary(id="c/d", pipeline_tag="robotics")])
    assert [m.id for m in client.trending(5)] == ["a/b"]


# ============================================================= download manager
class _FakeProcess:
    class ProcessState:
        NotRunning = 0

    def __init__(self, output: bytes = b"", running: bool = True) -> None:
        self._output = output
        self._running = running
        self.terminated = False

    def readAllStandardOutput(self) -> bytes:  # noqa: N802
        data, self._output = self._output, b""
        return data

    def readAllStandardError(self) -> bytes:  # noqa: N802
        return b"warning: slow mirror\n"

    def state(self) -> int:
        return 1 if self._running else 0

    def terminate(self) -> None:
        self.terminated = True
        self._running = False

    def waitForFinished(self, _ms: int) -> bool:  # noqa: N802
        return True

    def kill(self) -> None:
        self._running = False

    def deleteLater(self) -> None:  # noqa: N802
        pass


@pytest.fixture()
def manager(qapp: Any, tmp_path: Path) -> DownloadManager:
    return DownloadManager(Settings(tmp_path / "s.json"), ModelRegistry(tmp_path / "r.db"))


def test_stdout_parsing_and_finish(manager: DownloadManager, tmp_path: Path) -> None:
    job = DownloadJob("org/m", summary=ModelSummary(id="org/m", pipeline_tag="text-generation"))
    manager.jobs["org/m"] = job
    lines = (b'{"type":"plan","files":[{"name":"a.bin","size":10,"cached":false}],"total_bytes":10,'
             b'"download_bytes":10}\nnot json\n{"type":"progress","file":"a.bin","downloaded":5}\n{"type":"file_')
    proc = _FakeProcess(lines)
    manager._processes["org/m"] = proc
    updates: list[str] = []
    manager.job_updated.connect(updates.append)
    manager._on_stdout("org/m")
    assert job.files["a.bin"].downloaded == 5 and updates
    manager._on_stderr("org/m")
    assert job.log[-1] == "warning: slow mirror"
    proc._output = b'error","file":"a.bin","message":"boom"}\n'
    manager._on_stdout("org/m")
    assert "a.bin: boom" in job.log[-1]
    finished: list[tuple[str, bool]] = []
    manager.job_finished.connect(lambda repo, ok: finished.append((repo, ok)))
    proc._output = json.dumps({"type": "done", "path": str(tmp_path), "total_bytes": 10,
                               "files": ["a.bin"]}).encode() + b"\n"
    manager._on_finished("org/m", 0)
    assert job.status == "done" and finished == [("org/m", True)]
    assert manager.registry.get_model("org/m") is not None


def test_unexpected_exit_marks_error(manager: DownloadManager) -> None:
    job = DownloadJob("org/x", status="downloading")
    job.log.append("last words")
    manager.jobs["org/x"] = job
    manager._processes["org/x"] = _FakeProcess()
    manager._on_finished("org/x", 3)
    assert job.status == "error" and "last words" in (job.error or "")


def test_pause_resume_cancel_and_queue(manager: DownloadManager, monkeypatch: pytest.MonkeyPatch) -> None:
    launched: list[str] = []
    monkeypatch.setattr(manager, "_launch", lambda job: (launched.append(job.repo_id),
                                                         manager._processes.__setitem__(job.repo_id, _FakeProcess()),
                                                         setattr(job, "status", "downloading")))
    for repo in ("org/a", "org/b", "org/c"):
        manager.start(repo)
    assert launched == ["org/a", "org/b"] and manager.job("org/c").status == "queued"  # two at a time
    assert manager.start("org/a") is manager.job("org/a")  # already active: no duplicate
    proc = manager._processes["org/a"]
    manager.pause("org/a")
    assert proc.terminated and manager.job("org/a").status == "paused" and "org/a" in manager._intentional_stop
    manager._processes.pop("org/a")
    manager._pump()
    assert launched[-1] == "org/c"
    manager.resume("org/a")
    assert manager.job("org/a").status == "queued"
    manager.cancel("org/b")
    assert manager.job("org/b").status == "cancelled"
    assert manager.active_repos() >= {"org/c"}
    manager.shutdown()


def test_speed_and_eta() -> None:
    job = DownloadJob("org/s", download_bytes=1000)
    job.files = {"w": FileProgress("w", 1000, downloaded=0)}
    assert job.eta_s is None and job.fraction == 0.0
    job._last_time -= 1.0  # pretend a second passed
    job.files["w"].downloaded = 400
    DownloadManager._update_speed(None, job)  # type: ignore[arg-type]
    assert job.speed_bps > 0 and job.eta_s is not None and job.eta_s > 0


def test_registry_remaining_paths(tmp_path: Path) -> None:
    from noelle.app.services.model_registry import ModelRecord

    registry = ModelRegistry(tmp_path / "r.db")
    record = registry.upsert_model(ModelRecord("org/m", "text-generation", str(tmp_path), precision="fp16"))
    assert record.display_name == "m (fp16)" and record.api_name == "org/m:fp16"
    assert registry.get_model_by_id(record.id).repo_id == "org/m"
    registry.touch_model("org/m")
    assert registry.get_model("org/m").last_used_at is not None
    registry.set_trust_remote_code("org/m", True)
    assert registry.get_model("org/m").trust_remote_code
    assert registry.find_by_api_name("org/m:fp16").precision == "fp16"
    assert [r.repo_id for r in registry.iter_models()] == ["org/m"]
    registry.delete_adapter("none")
    assert registry.list_evaluations() == []
    assert registry.delete_model("org/m", "fp16", "hub") == 1
    registry.close()
