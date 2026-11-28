"""Local Training Studio: dataset normalization, planning, tracking and manager events."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from noelle.training.config import (  # noqa: E402
    ConfigError,
    DatasetConfig,
    HardwareInfo,
    LoRAConfig,
    ModelConfig,
    TrainingConfig,
    TrainingParams,
    auto_config,
    estimate_plan,
    load_recipe,
    save_recipe,
)
from noelle.training.dataset import analyze_dataset, prepare_dataset  # noqa: E402
from noelle.training.store import TrainingStore  # noqa: E402


def _rows(path: Path, rows: list[Any]) -> Path:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def _config(tmp_path: Path, **changes: Any) -> TrainingConfig:
    dataset = _rows(tmp_path / "data.jsonl", [
        {"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]},
        {"instruction": "Add 1 and 2", "output": "3"},
    ])
    config = TrainingConfig(
        name="demo", output_name="demo-lora", mode="lora",
        model=ModelConfig(id="org/tiny", local_path=str(tmp_path / "model")),
        dataset=DatasetConfig(path=str(dataset)),
        lora=LoRAConfig(), params=TrainingParams(),
    )
    for key, value in changes.items():
        setattr(config, key, value)
    return config


# ------------------------------------------------------------------ datasets


def test_analyze_dataset_normalizes_and_counts(tmp_path: Path) -> None:
    path = _rows(tmp_path / "data.jsonl", [
        {"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]},
        {"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]},
        {"instruction": "Add 1 and 2", "output": "3"},
        {"query": "What is Noelle?", "response": "A local AI studio."},
        {"text": "A plain-text training document."},
        {},
    ])
    report = analyze_dataset(path, max_seq_length=16, validation_ratio=0.5)
    assert report.rows == 6
    assert report.duplicates == 1
    assert report.invalid == 1
    assert report.valid == 4
    assert report.train + report.validation == report.valid
    assert report.train >= 1 and report.validation >= 1
    assert report.schema in ("chat", "instruction", "query", "text")
    assert report.sha256


def test_analyze_csv_and_markdown(tmp_path: Path) -> None:
    csv = tmp_path / "data.csv"
    csv.write_text("instruction,output\nSay hi,Hello\n", encoding="utf-8")
    report = analyze_dataset(csv)
    assert report.valid == 1 and report.schema == "instruction"

    md = tmp_path / "data.md"
    md.write_text("First paragraph.\n\nSecond paragraph.", encoding="utf-8")
    report = analyze_dataset(md)
    assert report.valid == 2 and report.schema == "text"


def test_prepare_dataset_writes_split_and_report(tmp_path: Path) -> None:
    path = _rows(tmp_path / "data.jsonl", [
        {"instruction": f"Question {i}", "output": f"Answer {i}"} for i in range(6)
    ])
    prepared = prepare_dataset(path, tmp_path / "prepared", validation_ratio=0.5, seed=7)
    train = [json.loads(line) for line in prepared.train_path.read_text(encoding="utf-8").splitlines()]
    validation = [json.loads(line) for line in prepared.validation_path.read_text(encoding="utf-8").splitlines()]
    assert len(train) == 3 and len(validation) == 3
    assert all("messages" in row for row in train + validation)
    report = json.loads(prepared.report_path.read_text(encoding="utf-8"))
    assert report["train"] == 3 and report["validation"] == 3


def test_prepare_dataset_rejects_empty(tmp_path: Path) -> None:
    path = _rows(tmp_path / "data.jsonl", [{}])
    with pytest.raises(ValueError, match="no valid rows"):
        prepare_dataset(path, tmp_path / "prepared")


def test_dependency_report_is_always_a_mapping() -> None:
    from noelle.training.trainer import dependency_report

    report = dependency_report()
    assert isinstance(report["versions"], dict)
    assert isinstance(report["missing"], list)
    assert isinstance(report["compatible"], bool)
    for name in ("torch", "transformers", "datasets", "accelerate", "peft", "trl"):
        assert name in report["versions"]


# ------------------------------------------------------------------- config


def test_config_roundtrip_yaml(tmp_path: Path) -> None:
    config = _config(tmp_path)
    path = save_recipe(config, tmp_path / "recipe.yaml")
    loaded = load_recipe(path)
    assert loaded.mode == "lora"
    assert loaded.model.id == "org/tiny"
    assert loaded.dataset.path == config.dataset.path
    assert loaded.lora.rank == 16
    assert loaded.params.epochs == 3.0


def test_config_validation_rejects_bad_values(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.dataset.path = str(tmp_path / "missing.jsonl")
    config.output_name = "../escape"
    config.params.epochs = 0
    with pytest.raises(ConfigError) as info:
        config.validate()
    text = str(info.value)
    assert "Dataset not found" in text
    assert "Output name" in text
    assert "Epochs" in text


def test_estimate_plan_marks_qlora_cpu_high_risk(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.mode = "qlora"
    plan = estimate_plan(config, HardwareInfo(device="cpu", ram_total_bytes=16 * 10**9),
                         param_count=1_000_000_000, dataset_samples=100, dataset_avg_tokens=256)
    assert plan.risk == "HIGH RISK"
    assert plan.quantization == "bitsandbytes-nf4"


def test_estimate_plan_small_gpu_recommends_qlora() -> None:
    config = TrainingConfig(name="x", output_name="x", mode="lora",
                            model=ModelConfig(id="org/model"), dataset=DatasetConfig(path="/tmp/data"))
    hardware = HardwareInfo(device="cuda", gpu_name="RTX 3060", vram_total_bytes=12 * 10**9,
                            vram_usable_bytes=12 * 10**9, ram_total_bytes=32 * 10**9)
    plan = estimate_plan(config, hardware, param_count=7_000_000_000, dataset_samples=1000,
                         dataset_avg_tokens=512)
    assert plan.risk in ("SAFE", "WARNING", "HIGH RISK")
    assert auto_config(config, hardware, 7_000_000_000)["mode"] in ("lora", "qlora")


# -------------------------------------------------------------------- store


def test_store_metrics_checkpoints_and_delete(tmp_path: Path) -> None:
    store = TrainingStore(tmp_path / "training.db")
    run = store.create(_config(tmp_path))
    store.add_metric(run["id"], {"step": 3, "total_steps": 10, "loss": 1.25, "learning_rate": 2e-4})
    store.add_metric(run["id"], {"step": 4, "total_steps": 10, "loss": 1.0})
    assert [item["step"] for item in store.metrics(run["id"])] == [3, 4]
    updated = store.get(run["id"])
    assert updated and updated["step"] == 4 and updated["loss"] == 1.0

    store.add_checkpoint(run["id"], tmp_path / "checkpoint-3", step=3, size_bytes=10)
    store.add_checkpoint(run["id"], tmp_path / "checkpoint-4", step=4, size_bytes=20)
    assert store.latest_checkpoint(run["id"])["step"] == 4
    assert store.delete(run["id"]) is True
    assert store.get(run["id"]) is None


# ------------------------------------------------------------------ manager


class _Registry:
    def __init__(self) -> None:
        self.adapters: list[Any] = []
        self.models: list[Any] = []

    def add_adapter(self, record: Any) -> Any:
        self.adapters.append(record)
        return record

    def upsert_model(self, record: Any) -> Any:
        self.models.append(record)
        return record


def _manager(store: TrainingStore, registry: _Registry) -> Any:
    from noelle.training.manager import TrainingManager

    settings = SimpleNamespace(get=lambda _key, default=None: default)
    return TrainingManager(settings, registry, store=store)


def test_manager_registers_adapter_after_completed_event(qtbot: Any, tmp_path: Path) -> None:
    store = TrainingStore(tmp_path / "training.db")
    registry = _Registry()
    manager = _manager(store, registry)
    config = _config(tmp_path)
    run = store.create(config)
    output = store.final_dir(run["id"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "adapter_model.safetensors").write_bytes(b"weights")

    manager._apply_event(run["id"], {"type": "training.progress", "step": 2, "loss": 0.5})
    assert store.metrics(run["id"]) == []  # the worker owns metric persistence

    manager._apply_event(run["id"], {"type": "training.completed", "output_path": str(output),
                                     "manifest": {"status": "completed"}})
    done = store.get(run["id"])
    assert done and done["status"] == "completed" and done["output_path"] == str(output)
    assert len(registry.adapters) == 1
    assert registry.adapters[0].base_model_id == "org/tiny"
    assert Path(registry.adapters[0].local_path) == output


def test_manager_registers_full_model_and_errors(qtbot: Any, tmp_path: Path) -> None:
    store = TrainingStore(tmp_path / "training.db")
    registry = _Registry()
    manager = _manager(store, registry)
    config = _config(tmp_path)
    config.mode = "full"
    run = store.create(config)
    output = store.final_dir(run["id"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text("{}", encoding="utf-8")
    manager._apply_event(run["id"], {"type": "training.completed", "output_path": str(output), "manifest": {}})
    assert len(registry.models) == 1
    assert registry.models[0].repo_id == f"trained/{config.output_name}"

    other = store.create(config)
    manager._apply_event(other["id"], {"type": "training.error", "code": "OOM", "message": "Out of memory"})
    failed = store.get(other["id"])
    assert failed and failed["status"] == "oom"
    assert failed["error"] == "Out of memory"


def test_manager_error_events_are_idempotent(qtbot: Any, tmp_path: Path) -> None:
    store = TrainingStore(tmp_path / "training.db")
    registry = _Registry()
    manager = _manager(store, registry)
    run = store.create(_config(tmp_path))
    finished: list[bool] = []
    manager.job_finished.connect(lambda _run_id, ok: finished.append(ok))
    manager._apply_event(run["id"], {"type": "training.error", "code": "OOM",
                                     "message": "Out of memory", "recommendations": ["Reduce batch size"]})
    manager._apply_event(run["id"], {"type": "training.error", "code": "OOM", "message": "Out of memory"})
    assert len(finished) == 1
    assert "Reduce batch size" in store.get(run["id"])["error"]


def test_manager_marks_leftover_runs_interrupted(qtbot: Any, tmp_path: Path) -> None:
    store = TrainingStore(tmp_path / "training.db")
    run = store.create(_config(tmp_path))
    store.update(run["id"], status="training")
    registry = _Registry()
    _manager(store, registry)
    assert store.get(run["id"])["status"] == "interrupted"


def test_manager_stop_uses_a_marker_file(qtbot: Any, tmp_path: Path) -> None:
    store = TrainingStore(tmp_path / "training.db")
    registry = _Registry()
    manager = _manager(store, registry)
    run = store.create(_config(tmp_path))
    store.update(run["id"], status="training")
    killed: list[bool] = []
    manager._processes[run["id"]] = SimpleNamespace(kill=lambda: killed.append(True))
    manager.stop(run["id"])
    assert store.get(run["id"])["status"] == "stopping"
    assert (store.run_dir(run["id"]) / "stop.requested").is_file()
    assert killed == []
    manager.stop(run["id"], force=True)
    assert killed == [True]


class _Settings:
    def __init__(self) -> None:
        self.values: dict[str, Any] = {}

    def get(self, key: str, default: Any = None) -> Any:
        return self.values.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self.values[key] = value


def test_training_page_lists_and_renders_runs(qtbot: Any, tmp_path: Path) -> None:
    from noelle.app.ui.pages.training_page import TrainingPage
    from noelle.runtime.preflight import MemoryBudget

    store = TrainingStore(tmp_path / "training.db")
    run = store.create(_config(tmp_path))
    registry = _Registry()

    def list_models(only_existing: bool = True) -> list[Any]:
        return []

    registry.list_models = list_models  # type: ignore[attr-defined]
    manager = _manager(store, registry)

    class Hardware:
        class Gpu:
            name = "CPU"
            total_memory = 0
            kind = "cpu"

        gpu = Gpu()

        @staticmethod
        def memory_budget(headroom: float = 0.0) -> MemoryBudget:
            return MemoryBudget(8 * 10**9, headroom, 16 * 10**9, "cpu")

    ctx = SimpleNamespace(
        settings=_Settings(), registry=registry, hardware=Hardware(), training=manager, extras={},
        toast=lambda *_a, **_k: None, open_settings=lambda: None, navigate=lambda _page: None,
    )
    page = TrainingPage(ctx)
    qtbot.addWidget(page)
    assert page.run_list.count() == 1
    assert run["id"] in page.run_list.item(0).text() or "demo" in page.run_list.item(0).text()
    page.run_list.setCurrentRow(0)
    assert page.run_title.text() == "demo"
    page.shutdown()