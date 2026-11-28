"""Local Training & Fine-Tuning Studio (MVP).

Pure-Python dataset/config/planner/store modules are safe to import from the desktop
shell. The heavy training stack (torch/transformers/TRL/PEFT) is imported only inside
the dedicated training worker.
"""

from .config import LoRAConfig, ModelConfig, TrainingConfig, TrainingParams, estimate_plan
from .dataset import DatasetReport, PreparedDataset, analyze_dataset, prepare_dataset
from .store import TrainingStore

__all__ = ["DatasetReport", "LoRAConfig", "ModelConfig", "PreparedDataset", "TrainingConfig", "TrainingParams",
           "TrainingStore", "analyze_dataset", "estimate_plan", "prepare_dataset"]