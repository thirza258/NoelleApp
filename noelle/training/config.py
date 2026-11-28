"""Training configuration, validation and hardware planning (pure Python)."""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

MODES = ("lora", "qlora", "full", "continued_pretrain")
RISK_SAFE = "SAFE"
RISK_WARNING = "WARNING"
RISK_HIGH = "HIGH RISK"
RISK_UNSUPPORTED = "UNSUPPORTED"
DEFAULT_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")


class ConfigError(ValueError):
    def __init__(self, problems: list[str]) -> None:
        super().__init__("; ".join(problems))
        self.problems = problems


@dataclass
class ModelConfig:
    id: str
    revision: str | None = None
    local_path: str | None = None
    tokenizer: str | None = None
    trust_remote_code: bool = False

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ModelConfig:
        return cls(id=str(data.get("id") or data.get("model") or ""), revision=data.get("revision") or None,
                   local_path=data.get("local_path") or None, tokenizer=data.get("tokenizer") or None,
                   trust_remote_code=bool(data.get("trust_remote_code", False)))


@dataclass
class DatasetConfig:
    path: str
    validation_ratio: float = 0.1
    max_seq_length: int = 1024
    dedupe: bool = True
    drop_invalid: bool = True
    trim_long: bool = False
    prepared_dir: str | None = None
    sha256: str | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DatasetConfig:
        return cls(path=str(data.get("path") or ""), validation_ratio=float(data.get("validation_ratio", 0.1)),
                   max_seq_length=int(data.get("max_seq_length", 1024)), dedupe=bool(data.get("dedupe", True)),
                   drop_invalid=bool(data.get("drop_invalid", True)), trim_long=bool(data.get("trim_long", False)),
                   prepared_dir=data.get("prepared_dir") or None, sha256=data.get("sha256") or None)


@dataclass
class LoRAConfig:
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: list(DEFAULT_TARGETS))
    bias: str = "none"
    task_type: str = "CAUSAL_LM"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> LoRAConfig:
        targets = data.get("target_modules")
        return cls(rank=int(data.get("rank", 16)), alpha=int(data.get("alpha", 32)),
                   dropout=float(data.get("dropout", 0.05)),
                   target_modules=[str(item) for item in targets] if isinstance(targets, list) else list(DEFAULT_TARGETS),
                   bias=str(data.get("bias", "none")), task_type=str(data.get("task_type", "CAUSAL_LM")))


@dataclass
class TrainingParams:
    epochs: float = 3.0
    learning_rate: float = 2e-4
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    scheduler: str = "cosine"
    optimizer: str = "auto"
    precision: str = "auto"  # auto | fp32 | fp16 | bf16 | nf4 | int8
    gradient_checkpointing: bool = True
    seed: int = 42
    save_steps: int = 100
    logging_steps: int = 1
    save_total_limit: int = 3

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TrainingParams:
        return cls(epochs=float(data.get("epochs", 3.0)), learning_rate=float(data.get("learning_rate", 2e-4)),
                   batch_size=int(data.get("batch_size", 1)),
                   gradient_accumulation_steps=int(data.get("gradient_accumulation_steps", 8)),
                   warmup_ratio=float(data.get("warmup_ratio", 0.03)),
                   weight_decay=float(data.get("weight_decay", 0.0)),
                   scheduler=str(data.get("scheduler", "cosine")), optimizer=str(data.get("optimizer", "auto")),
                   precision=str(data.get("precision", "auto")),
                   gradient_checkpointing=bool(data.get("gradient_checkpointing", True)),
                   seed=int(data.get("seed", 42)), save_steps=int(data.get("save_steps", 100)),
                   logging_steps=int(data.get("logging_steps", 1)),
                   save_total_limit=int(data.get("save_total_limit", 3)))


@dataclass
class TrainingConfig:
    name: str
    output_name: str
    mode: str = "qlora"
    model: ModelConfig = field(default_factory=lambda: ModelConfig(id=""))
    dataset: DatasetConfig = field(default_factory=lambda: DatasetConfig(path=""))
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    params: TrainingParams = field(default_factory=TrainingParams)
    resume_from: str | None = None
    seed: int = 42

    @property
    def effective_batch(self) -> int:
        return max(self.params.batch_size * self.params.gradient_accumulation_steps, 1)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TrainingConfig:
        model = data.get("model") if isinstance(data.get("model"), Mapping) else {"id": data.get("model")}
        dataset = data.get("dataset") if isinstance(data.get("dataset"), Mapping) else {"path": data.get("dataset")}
        name = str(data.get("name") or data.get("output_name") or "training-run")
        return cls(
            name=name, output_name=str(data.get("output_name") or name), mode=str(data.get("mode") or "qlora").lower(),
            model=ModelConfig.from_dict(model or {}), dataset=DatasetConfig.from_dict(dataset or {}),
            lora=LoRAConfig.from_dict(data.get("lora") if isinstance(data.get("lora"), Mapping) else {}),
            params=TrainingParams.from_dict(data.get("training") if isinstance(data.get("training"), Mapping)
                                            else data.get("params") if isinstance(data.get("params"), Mapping) else {}),
            resume_from=data.get("resume_from") or None, seed=int(data.get("seed", 42)),
        )

    @classmethod
    def from_yaml(cls, text: str) -> TrainingConfig:
        import yaml

        data = yaml.safe_load(text)
        if not isinstance(data, Mapping):
            raise ConfigError(["Recipe must be a YAML/JSON object."])
        return cls.from_dict(data)

    @classmethod
    def from_file(cls, path: str | Path) -> TrainingConfig:
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() in (".yaml", ".yml"):
            return cls.from_yaml(text)
        return cls.from_dict(json.loads(text))

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["effective_batch"] = self.effective_batch
        return data

    def to_yaml(self) -> str:
        import yaml

        return yaml.safe_dump(self.to_dict(), sort_keys=False, allow_unicode=True)

    def validate(self) -> None:
        problems: list[str] = []
        if not self.model.id and not self.model.local_path:
            problems.append("Choose a base model or local model folder.")
        if not self.dataset.path:
            problems.append("Choose a dataset file.")
        elif not Path(self.dataset.path).expanduser().is_file():
            problems.append(f"Dataset not found: {self.dataset.path}")
        if self.mode not in MODES:
            problems.append(f"Unknown training mode '{self.mode}'.")
        if self.params.epochs <= 0:
            problems.append("Epochs must be greater than zero.")
        if self.params.learning_rate <= 0:
            problems.append("Learning rate must be greater than zero.")
        if self.params.batch_size < 1 or self.params.gradient_accumulation_steps < 1:
            problems.append("Batch size and gradient accumulation must be at least 1.")
        if self.dataset.max_seq_length < 128:
            problems.append("Sequence length must be at least 128 tokens.")
        if self.mode in ("lora", "qlora"):
            if self.lora.rank < 1 or self.lora.rank > 512:
                problems.append("LoRA rank must be between 1 and 512.")
            if self.lora.alpha < 1:
                problems.append("LoRA alpha must be at least 1.")
            if not 0.0 <= self.lora.dropout < 1.0:
                problems.append("LoRA dropout must be in [0, 1).")
            if not self.lora.target_modules:
                problems.append("Choose at least one LoRA target module.")
        if not _SAFE_NAME.fullmatch(self.output_name):
            problems.append("Output name may contain letters, digits, '.', '_' and '-' (max 80).")
        if problems:
            raise ConfigError(problems)


@dataclass(frozen=True)
class HardwareInfo:
    device: str = "cpu"
    gpu_name: str = ""
    vram_total_bytes: int = 0
    vram_usable_bytes: int = 0
    ram_total_bytes: int = 0
    disk_free_bytes: int = 0

    @property
    def cuda(self) -> bool:
        return self.device == "cuda"


@dataclass
class TrainingPlan:
    mode: str
    precision: str
    quantization: str | None
    batch_size: int
    gradient_accumulation_steps: int
    max_seq_length: int
    estimated_vram_bytes: int | None
    estimated_ram_bytes: int
    disk_bytes: int
    risk: str
    expected_steps: int
    estimated_seconds: int
    notes: list[str] = field(default_factory=list)
    recommended: dict[str, Any] = field(default_factory=dict)

    @property
    def effective_batch(self) -> int:
        return max(self.batch_size * self.gradient_accumulation_steps, 1)

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "effective_batch": self.effective_batch}


def _precision_for(mode: str, precision: str, device: str) -> tuple[str, str | None]:
    if mode == "qlora":
        return "nf4", "bitsandbytes-nf4"
    if mode == "full":
        if device == "cpu":
            return "fp32", None
        return (precision if precision in ("bf16", "fp16", "fp32") else "bf16"), None
    if device == "cuda":
        return (precision if precision in ("bf16", "fp16", "fp32") else "bf16"), None
    return "fp32", None


def _weights_per_param(mode: str, precision: str) -> float:
    if mode == "qlora":
        return 0.55  # 4-bit weights + quantization scales
    return 4.0 if precision == "fp32" else 2.0


def _throughput(params: int, mode: str, device: str) -> float:
    base = 4000.0 if params <= 1e9 else 1800.0 if params <= 2e9 else 600.0 if params <= 8e9 else 220.0
    if device == "cpu":
        base *= 0.03
    if mode == "qlora":
        base *= 0.8
    return max(base, 1.0)


def estimate_plan(config: TrainingConfig, hardware: HardwareInfo, *, param_count: int | None = None,
                  dataset_samples: int = 0, dataset_avg_tokens: int = 0) -> TrainingPlan:
    params = max(int(param_count or 0), 0)
    precision, quantization = _precision_for(config.mode, config.params.precision, hardware.device)
    notes: list[str] = []
    if params == 0:
        notes.append("Parameter count is unknown; memory estimates use a conservative placeholder.")
        params = 1_000_000_000

    weight_bytes = params * _weights_per_param(config.mode, precision)
    trainable_ratio = config.lora.rank / 64 if config.mode in ("lora", "qlora") else 1.0
    trainable_params = max(params * min(trainable_ratio * 0.02, 0.05), 1) if config.mode in ("lora", "qlora") else params
    if config.mode == "full":
        optimizer_bytes = params * (12.0 if hardware.device != "cpu" else 16.0)
    else:
        optimizer_bytes = trainable_params * 16.0
    seq = config.dataset.max_seq_length
    hidden = max(int(math.sqrt(params / 12 / 32)), 512)
    activation_bytes = config.params.batch_size * seq * hidden * 32 * 4
    if config.params.gradient_checkpointing:
        activation_bytes *= 0.55
    overhead = 1_000_000_000
    vram = int(weight_bytes + optimizer_bytes + activation_bytes + overhead) if hardware.device != "cpu" else 0
    ram = int(weight_bytes * 1.2 + (dataset_samples * dataset_avg_tokens * 4) + 512_000_000)
    disk = int(weight_bytes + config.params.save_total_limit * weight_bytes * 0.6 + 100_000_000)

    samples = max(dataset_samples, 1)
    steps = max(int(math.ceil(samples / max(config.effective_batch, 1) * config.params.epochs)), 1)
    total_tokens = samples * max(dataset_avg_tokens, 32) * config.params.epochs
    seconds = int(total_tokens / _throughput(params, config.mode, hardware.device))

    usable = hardware.vram_usable_bytes or hardware.vram_total_bytes
    if hardware.device == "cpu":
        risk = RISK_WARNING
        notes.append("CPU training is extremely slow; use a small model and a short dataset.")
    elif usable <= 0:
        risk = RISK_WARNING
        notes.append("GPU memory could not be measured.")
    elif vram <= usable * 0.70:
        risk = RISK_SAFE
    elif vram <= usable * 0.95:
        risk = RISK_WARNING
    elif vram <= usable * 1.35:
        risk = RISK_HIGH
    else:
        risk = RISK_UNSUPPORTED
    if config.mode == "qlora" and not hardware.cuda:
        risk = RISK_HIGH
        notes.append("QLoRA uses bitsandbytes NF4 and requires an NVIDIA CUDA GPU.")
    if config.mode == "full" and risk in (RISK_SAFE, RISK_WARNING):
        notes.append("Full fine-tuning stores full optimizer states; keep a fallback checkpoint.")
    if config.mode == "continued_pretrain":
        notes.append("Continued pretraining is advanced: validate on held-out domain text and watch for forgetting.")

    recommended = auto_config(config, hardware, params, dataset_samples, dataset_avg_tokens)
    return TrainingPlan(config.mode, precision, quantization, config.params.batch_size,
                        config.params.gradient_accumulation_steps, seq, vram or None, ram, disk, risk, steps,
                        seconds, notes, recommended)


def auto_config(config: TrainingConfig, hardware: HardwareInfo, param_count: int,
                dataset_samples: int = 0, dataset_avg_tokens: int = 0) -> dict[str, Any]:
    """Recommend a conservative configuration for the current GPU."""
    usable = hardware.vram_usable_bytes or hardware.vram_total_bytes
    params = max(param_count, 1)
    if hardware.cuda and usable and usable < 12_000_000_000:
        mode = "qlora"
    elif hardware.cuda:
        mode = "lora" if params > 2_000_000_000 or not usable else "full"
    elif hardware.device == "mps":
        mode = "lora"
    else:
        mode = "lora"  # QLoRA is CUDA/bitsandbytes-only
    max_seq = 1024 if usable >= 8_000_000_000 else 512
    batch = 1
    grad = 16 if usable < 8_000_000_000 else 8
    target = config.dataset.max_seq_length
    if target > max_seq:
        approach = {"max_seq_length": max_seq, "batch_size": batch, "gradient_accumulation_steps": grad,
                    "precision": "bf16" if hardware.device == "cuda" else "fp32", "mode": mode,
                    "note": f"Reduce sequence length from {target} to {max_seq}."}
    else:
        approach = {"max_seq_length": target, "batch_size": batch, "gradient_accumulation_steps": grad,
                    "precision": "bf16" if hardware.device == "cuda" else "fp32", "mode": mode,
                    "note": "Conservative configuration for this GPU."}
    return approach


def load_recipe(path: str | Path) -> TrainingConfig:
    return TrainingConfig.from_file(path)


def save_recipe(config: TrainingConfig, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(config.to_yaml(), encoding="utf-8")
    return path