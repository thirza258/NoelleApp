"""Training worker backend: dataset preparation, LoRA/QLoRA/full SFT and checkpoints.

Every heavy import happens inside this module's functions so the desktop shell and the
inference worker never import TRL, PEFT or bitsandbytes.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import logging
import os
import platform
import time
from pathlib import Path
from typing import Any, Callable, Mapping

from noelle import __version__
from noelle.training.config import ConfigError, TrainingConfig
from noelle.training.dataset import PreparedDataset, prepare_dataset

log = logging.getLogger("noelle.training")

Emit = Callable[[dict[str, Any]], None]


class TrainingDependencyError(RuntimeError):
    def __init__(self, missing: list[str], versions: Mapping[str, Any]) -> None:
        super().__init__("Training dependencies are missing: " + ", ".join(missing))
        self.missing = missing
        self.versions = versions


def dependency_report() -> dict[str, Any]:
    versions: dict[str, str | None] = {}
    missing: list[str] = []
    for distribution, module in (("torch", "torch"), ("transformers", "transformers"), ("datasets", "datasets"),
                                 ("accelerate", "accelerate"), ("peft", "peft"), ("trl", "trl")):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
            missing.append(distribution)
        except Exception:  # noqa: BLE001
            versions[distribution] = None
    try:
        bitsandbytes = importlib.util.find_spec("bitsandbytes") is not None
    except Exception:  # noqa: BLE001
        bitsandbytes = False
    versions["bitsandbytes"] = "installed" if bitsandbytes else None
    return {"versions": versions, "missing": missing, "bitsandbytes": bitsandbytes,
            "compatible": not missing}


def _versions() -> dict[str, Any]:
    report = dependency_report()
    data: dict[str, Any] = {**report["versions"], "python": platform.python_version(),
                            "platform": platform.platform(), "noelle": __version__}
    try:
        import torch

        data["cuda"] = getattr(torch.version, "cuda", None)
        data["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            data["gpu"] = torch.cuda.get_device_name(0)
            data["gpu_capability"] = list(torch.cuda.get_device_capability(0))
    except Exception:  # noqa: BLE001
        pass
    return data


def _report_dict(report: Any) -> dict[str, Any]:
    """The manifest and worker events carry counts and hashes, never dataset samples."""
    data = report.to_dict()
    data.pop("preview", None)
    return data


def _prepare(config: TrainingConfig, emit: Emit, *, tokenizer: Any = None) -> PreparedDataset:
    emit({"type": "training.preparing", "stage": "dataset"})
    output = Path(config.dataset.prepared_dir or Path(config.dataset.path).parent / ".noelle-prepared")
    prepared = prepare_dataset(config.dataset.path, output, tokenizer=tokenizer,
                               max_seq_length=config.dataset.max_seq_length,
                               validation_ratio=config.dataset.validation_ratio, dedupe=config.dataset.dedupe,
                               drop_invalid=config.dataset.drop_invalid, trim_long=config.dataset.trim_long,
                               seed=config.seed)
    emit({"type": "training.dataset", "report": _report_dict(prepared.report)})
    return prepared


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _conversation_dataset(rows: list[dict[str, Any]]) -> Any:
    from datasets import Dataset

    if all(isinstance(row.get("messages"), list) for row in rows):
        return Dataset.from_list([{"messages": row["messages"]} for row in rows])
    if all(isinstance(row.get("text"), str) for row in rows):
        return Dataset.from_list([{"text": row["text"]} for row in rows])
    # Mixed chat and plain-text rows need one uniform column for SFTTrainer.
    flat: list[dict[str, str]] = []
    for row in rows:
        if isinstance(row.get("messages"), list):
            text = "\n".join(f"{message.get('role')}: {message.get('content')}" for message in row["messages"])
        else:
            text = str(row.get("text") or "")
        flat.append({"text": text})
    return Dataset.from_list(flat)


def _lora_targets(model: Any, requested: list[str]) -> list[str]:
    names = {name.rsplit(".", 1)[-1] for name, _module in model.named_modules()}
    selected = [name for name in requested if name in names]
    return selected


def _require_dependencies(config: TrainingConfig, run_id: str, store: Any, emit: Emit) -> dict[str, Any]:
    report = dependency_report()
    required = ["torch", "transformers", "datasets", "accelerate", "peft", "trl"]
    missing = [name for name in required if not report["versions"].get(name)]
    if config.mode == "qlora":
        if not report["bitsandbytes"]:
            emit({"type": "training.error", "code": "DEPENDENCY_MISSING",
                  "message": "QLoRA needs bitsandbytes with an NVIDIA CUDA GPU.",
                  "missing": ["bitsandbytes"], "versions": report["versions"]})
            store.update(run_id, status="failed", error="bitsandbytes is not installed.", finished_at=time.time())
            raise TrainingDependencyError(["bitsandbytes"], report["versions"])
    if missing:
        emit({"type": "training.error", "code": "DEPENDENCY_MISSING",
              "message": "Training dependencies are missing: " + ", ".join(missing),
              "missing": missing, "versions": report["versions"]})
        store.update(run_id, status="failed", error="Missing: " + ", ".join(missing), finished_at=time.time())
        raise TrainingDependencyError(missing, report["versions"])
    return report


def _stop_requested(stop_event: Any) -> bool:
    try:
        return bool(stop_event.is_set())
    except AttributeError:
        return False


def _abort_if_stopped(stop_event: Any, run_id: str, store: Any, emit: Emit) -> bool:
    if not _stop_requested(stop_event):
        return False
    store.update(run_id, status="stopped", finished_at=time.time())
    emit({"type": "training.stopped", "run_id": run_id})
    return True


def run_training(config: TrainingConfig, run_id: str, store: Any, emit: Emit,
                 stop_event: Any) -> dict[str, Any]:
    """Run one training job. Emits JSON-serialisable events; returns the final manifest."""
    config.validate()
    store.update(run_id, status="preparing", started_at=time.time())
    emit({"type": "training.started", "run_id": run_id, "mode": config.mode})
    _require_dependencies(config, run_id, store, emit)

    # 1. Tokenizer is needed for token-accurate dataset statistics.
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # dependency_report should already have caught this
        emit({"type": "training.error", "code": "DEPENDENCY_MISSING",
              "message": "Training dependencies are not installed in this environment.",
              "detail": str(exc)})
        store.update(run_id, status="failed", error="Missing training dependencies.", finished_at=time.time())
        raise TrainingDependencyError(["transformers"], {}) from exc
    tokenizer_source = config.model.tokenizer or config.model.local_path or config.model.id
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, revision=config.model.revision,
                                              trust_remote_code=config.model.trust_remote_code)
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    emit({"type": "training.preparing", "stage": "tokenizer"})
    if _abort_if_stopped(stop_event, run_id, store, emit):
        return {"run_id": run_id, "status": "stopped"}

    # 2. Dataset pipeline (artifacts stay inside the run folder).
    config.dataset.prepared_dir = str(store.dataset_dir(run_id))
    prepared = _prepare(config, emit, tokenizer=tokenizer)
    store.update(run_id, dataset_sha=prepared.report.sha256, status="preparing")
    if _abort_if_stopped(stop_event, run_id, store, emit):
        return {"run_id": run_id, "status": "stopped"}
    train_rows = _load_rows(prepared.train_path)
    validation_rows = _load_rows(prepared.validation_path)
    if not train_rows:
        emit({"type": "training.error", "code": "EMPTY_DATASET", "message": "The training split is empty."})
        store.update(run_id, status="failed", error="Empty training split.", finished_at=time.time())
        raise ConfigError(["The training split is empty."])
    samples = len(train_rows)
    avg_tokens = int(prepared.report.avg_tokens or 0)

    # 3. Hardware plan (recorded in the manifest).
    from noelle.training.hardware import hardware_info
    from noelle.training.config import estimate_plan as plan_fn

    hardware = hardware_info()
    param_count = None
    try:
        from noelle.app.services.library_import import count_safetensors_params
        from noelle.runtime.hub_cache import cached_snapshot_path

        snapshot = (Path(config.model.local_path) if config.model.local_path
                    else cached_snapshot_path(config.model.id, config.model.revision))
        if snapshot is not None:
            param_count = count_safetensors_params(Path(snapshot))
    except Exception:  # noqa: BLE001
        param_count = None
    plan = plan_fn(config, hardware, param_count=param_count, dataset_samples=samples, dataset_avg_tokens=avg_tokens)
    emit({"type": "training.plan", "plan": plan.to_dict()})
    if plan.risk == "UNSUPPORTED":
        emit({"type": "training.warning", "message": "The plan is estimated to exceed available memory.",
              "plan": plan.to_dict()})
    if _abort_if_stopped(stop_event, run_id, store, emit):
        return {"run_id": run_id, "status": "stopped"}

    # 4. Training dependencies were verified before any model or network access.

    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig, TrainerCallback, TrainingArguments
    from trl import SFTTrainer

    if config.mode == "qlora" and not torch.cuda.is_available():
        message = "QLoRA needs an NVIDIA CUDA GPU; bitsandbytes 4-bit training is not available here."
        emit({"type": "training.error", "code": "QLORA_DEVICE", "message": message})
        store.update(run_id, status="failed", error=message, finished_at=time.time())
        raise ConfigError([message])

    source = config.model.local_path or config.model.id
    common: dict[str, Any] = {"trust_remote_code": config.model.trust_remote_code}
    if config.model.revision and not config.model.local_path:
        common["revision"] = config.model.revision
    dtype = torch.bfloat16 if plan.precision == "bf16" and torch.cuda.is_bf16_supported() else torch.float16
    if plan.precision == "nf4":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if plan.precision == "fp32":
        dtype = torch.float32
    model_kwargs: dict[str, Any] = {**common, "device_map": "auto" if hardware.cuda else None}
    if model_kwargs["device_map"] is None:
        model_kwargs.pop("device_map")
    if config.mode == "qlora":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype)
    else:
        model_kwargs["torch_dtype"] = dtype
    emit({"type": "training.preparing", "stage": "model"})
    model = AutoModelForCausalLM.from_pretrained(source, **model_kwargs)
    if config.mode == "qlora":
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=config.params.gradient_checkpointing)
    elif config.params.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    peft_config = None
    if config.mode in ("lora", "qlora"):
        targets = _lora_targets(model, config.lora.target_modules)
        if not targets:
            emit({"type": "training.error", "code": "LORA_TARGETS",
                  "message": "None of the selected LoRA target modules exist in this model."})
            raise ConfigError(["No matching LoRA target modules."])
        peft_config = LoraConfig(r=config.lora.rank, lora_alpha=config.lora.alpha,
                                 lora_dropout=config.lora.dropout, target_modules=targets,
                                 bias=config.lora.bias, task_type=config.lora.task_type)
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        if config.params.gradient_checkpointing and hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    train_dataset = _conversation_dataset(train_rows)
    eval_dataset = _conversation_dataset(validation_rows) if validation_rows else None

    output_dir = str(store.run_dir(run_id))
    args_kwargs: dict[str, Any] = {
        "output_dir": output_dir, "num_train_epochs": config.params.epochs,
        "per_device_train_batch_size": config.params.batch_size,
        "gradient_accumulation_steps": config.params.gradient_accumulation_steps,
        "learning_rate": config.params.learning_rate, "warmup_ratio": config.params.warmup_ratio,
        "weight_decay": config.params.weight_decay, "lr_scheduler_type": config.params.scheduler,
        "logging_steps": max(config.params.logging_steps, 1), "save_steps": max(config.params.save_steps, 1),
        "save_total_limit": max(config.params.save_total_limit, 1), "seed": config.seed,
        "report_to": [], "remove_unused_columns": False,
        "fp16": dtype == torch.float16, "bf16": dtype == torch.bfloat16,
        "gradient_checkpointing": config.params.gradient_checkpointing,
    }
    optimizer = config.params.optimizer
    if optimizer == "auto":
        optimizer = "paged_adamw_8bit" if config.mode == "qlora" else "adamw_torch"
    args_kwargs["optim"] = optimizer
    try:
        training_args = TrainingArguments(**args_kwargs)
    except TypeError as exc:
        emit({"type": "training.error", "code": "TRAINING_ARGS",
              "message": f"This Transformers version rejected a training argument: {exc}"})
        raise
    if hasattr(training_args, "max_length"):
        training_args.max_length = config.dataset.max_seq_length

    class ProgressCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):  # type: ignore[no-untyped-def]
            logs = logs or {}
            payload = {"step": state.global_step, "total_steps": state.max_steps,
                       "epoch": state.epoch, "loss": logs.get("loss"),
                       "learning_rate": logs.get("learning_rate"),
                       "tokens_per_second": logs.get("train_tokens_per_second"),
                       "gpu_memory_mb": logs.get("gpu_memory_mb") or _gpu_memory_mb()}
            store.add_metric(run_id, payload)
            emit({"type": "training.progress", "run_id": run_id, **payload})

        def on_step_end(self, args, state, control, **kwargs):  # type: ignore[no-untyped-def]
            if stop_event.is_set():
                control.should_training_stop = True
                control.should_save = True
                emit({"type": "training.warning", "message": "Stop requested; saving a checkpoint at the next step."})

        def on_save(self, args, state, control, **kwargs):  # type: ignore[no-untyped-def]
            folder = Path(args.output_dir) / f"checkpoint-{state.global_step}"
            if folder.is_dir():
                size = sum(p.stat().st_size for p in folder.rglob("*") if p.is_file())
                store.add_checkpoint(run_id, folder, step=state.global_step, size_bytes=size)
                emit({"type": "training.checkpoint", "path": str(folder), "step": state.global_step,
                      "size_bytes": size})

    trainer = _build_sft_trainer(SFTTrainer, model, training_args, train_dataset, eval_dataset, peft_config,
                                 tokenizer, config.dataset.max_seq_length, ProgressCallback)
    store.update(run_id, status="training", pid=os.getpid())
    resume = config.resume_from or None
    if resume and not Path(resume).exists():
        emit({"type": "training.warning", "message": f"Resume checkpoint not found: {resume}"})
        resume = None
    try:
        trainer.train(resume_from_checkpoint=resume)
    except Exception as exc:  # noqa: BLE001
        oom = "out of memory" in str(exc).lower() or "cuda error: out of memory" in str(exc).lower()
        code = "OOM" if oom else "TRAINING_ERROR"
        message = ("Training ran out of GPU memory." if oom else f"Training failed: {exc}")
        recommendations = (["Reduce sequence length", "Reduce batch size", "Enable gradient checkpointing",
                            "Use QLoRA", "Reduce LoRA rank"] if oom else [])
        emit({"type": "training.error", "code": code, "message": message, "detail": str(exc),
              "recommendations": recommendations, "plan": plan.to_dict()})
        store.update(run_id, status="oom" if oom else "failed", error=message, finished_at=time.time())
        _release(model)
        raise
    final = store.final_dir(run_id)
    final.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final))
    tokenizer.save_pretrained(str(final))
    stopped = stop_event.is_set()
    manifest = {
        "run_id": run_id, "status": "stopped" if stopped else "completed", "config": config.to_dict(),
        "dataset": {"path": config.dataset.path, "sha256": prepared.report.sha256,
                    "train_rows": len(train_rows), "validation_rows": len(validation_rows),
                    "report": _report_dict(prepared.report)},
        "plan": plan.to_dict(), "versions": _versions(), "hardware": _hardware_dict(hardware),
        "output": str(final), "checkpoints": [str(item["path"]) for item in store.checkpoints(run_id)],
        "finished_at": time.time(),
    }
    (store.run_dir(run_id) / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str),
                                                        encoding="utf-8")
    if stopped:
        store.update(run_id, status="stopped", output_path=str(final),
                     manifest_json=json.dumps(manifest, default=str), finished_at=time.time())
        emit({"type": "training.stopped", "run_id": run_id, "output_path": str(final), "manifest": manifest})
        _release(model)
        return manifest
    store.update(run_id, status="completed", output_path=str(final), manifest_json=json.dumps(manifest, default=str),
                 finished_at=time.time())
    emit({"type": "training.completed", "run_id": run_id, "output_path": str(final), "manifest": manifest})
    _release(model)
    return manifest


def _build_sft_trainer(trainer_class: Any, model: Any, training_args: Any, train_dataset: Any,
                       eval_dataset: Any, peft_config: Any, tokenizer: Any, max_seq_length: int,
                       callback_class: Any) -> Any:
    """Bridge TRL API changes: SFTConfig (new) vs TrainingArguments + max_seq_length (older)."""
    common = {"model": model, "train_dataset": train_dataset, "eval_dataset": eval_dataset,
              "peft_config": peft_config, "callbacks": [callback_class()]}
    try:
        from trl import SFTConfig

        config = SFTConfig(**{key: value for key, value in vars(training_args).items()
                              if not key.startswith("_") and value is not None}, max_length=max_seq_length)
        return trainer_class(**common, args=config, processing_class=tokenizer)
    except Exception:  # noqa: BLE001 - older TRL or versions without SFTConfig
        pass
    try:
        return trainer_class(**common, args=training_args, processing_class=tokenizer, max_length=max_seq_length)
    except TypeError:
        pass
    try:
        return trainer_class(**common, args=training_args, processing_class=tokenizer,
                             max_seq_length=max_seq_length)
    except TypeError:
        return trainer_class(**common, args=training_args, tokenizer=tokenizer,
                             max_seq_length=max_seq_length)


def _hardware_dict(hardware: Any) -> dict[str, Any]:
    return {"device": hardware.device, "gpu": hardware.gpu_name, "vram_total_bytes": hardware.vram_total_bytes,
            "vram_usable_bytes": hardware.vram_usable_bytes, "ram_total_bytes": hardware.ram_total_bytes}


def _gpu_memory_mb() -> float | None:
    try:
        import torch

        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / 1024 / 1024, 1)
    except Exception:  # noqa: BLE001
        pass
    return None


def _release(model: Any) -> None:
    try:
        del model
        import gc

        gc.collect()
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def merge_adapter(base: str, adapter: str, output: str, *, trust_remote_code: bool = False) -> str:
    """Merge a LoRA adapter into a full checkpoint (used by the UI/CLI export action)."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float16,
                                                 trust_remote_code=trust_remote_code)
    merged = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    target = Path(output)
    target.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(target))
    AutoTokenizer.from_pretrained(base, trust_remote_code=trust_remote_code).save_pretrained(str(target))
    return str(target)