"""Hardware snapshot for the training planner (no Qt imports: used by the training worker)."""

from __future__ import annotations

import shutil
from pathlib import Path

from noelle import paths
from noelle.runtime import devices
from noelle.training.config import HardwareInfo


def _free_disk(path: Path) -> int:
    try:
        return int(shutil.disk_usage(path).free)
    except OSError:
        return 0


def hardware_info() -> HardwareInfo:
    device = devices.resolve_device("auto")
    budget = devices.memory_budget(device, None, 0.0)
    snap = devices.memory_snapshot(device)
    gpu_name = str(snap.get("device_name") or devices.device_name(device))
    return HardwareInfo(device=device, gpu_name=gpu_name,
                        vram_total_bytes=int(budget.total_bytes if device in ("cuda", "mps") else 0),
                        vram_usable_bytes=int(budget.usable_bytes if device in ("cuda", "mps") else 0),
                        ram_total_bytes=int(budget.system_ram_bytes or 0),
                        disk_free_bytes=_free_disk(paths.home()))