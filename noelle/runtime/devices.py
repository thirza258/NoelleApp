"""Device selection, dtype policy and memory accounting for the worker.

Imports torch, so it must only be used inside worker processes.
"""

from __future__ import annotations

import gc
import os
import platform
import subprocess
from functools import lru_cache
from typing import Any

import psutil
import torch

from noelle.runtime.preflight import DEFAULT_HEADROOM, GB, MemoryBudget


def resolve_device(preference: str | None = "auto") -> str:
    pref = (preference or "auto").lower()
    cuda = torch.cuda.is_available()
    mps = torch.backends.mps.is_available()
    if pref in ("cuda", "gpu") and cuda:
        return "cuda"
    if pref in ("mps", "metal") and mps:
        return "mps"
    if pref == "cpu":
        return "cpu"
    if cuda:
        return "cuda"
    if mps:
        return "mps"
    return "cpu"


@lru_cache(maxsize=4)
def device_name(device: str) -> str:
    if device == "cuda":
        return torch.cuda.get_device_name(0)
    if device == "mps":
        try:
            out = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True,
                                 timeout=2).stdout.strip()
            return f"{out} GPU" if out else "Apple Silicon GPU"
        except (OSError, subprocess.SubprocessError):
            return "Apple Silicon GPU"
    return platform.processor() or platform.machine() or "CPU"


def torch_dtype(device: str, precision: str | None) -> torch.dtype:
    precision = (precision or "auto").lower()
    if precision == "fp32":
        return torch.float32
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16 if device != "cpu" else torch.float32
    if precision == "auto":
        if device == "cuda":
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        if device == "mps":
            return torch.float16
        return torch.float32
    # quantized precisions compute in half precision on accelerators
    return torch.float16 if device != "cpu" else torch.float32


def memory_snapshot(device: str) -> dict[str, Any]:
    vm = psutil.virtual_memory()
    snap: dict[str, Any] = {
        "device": device,
        "device_name": device_name(device),
        "system_total": vm.total,
        "system_available": vm.available,
        "process_rss": psutil.Process().memory_info().rss,
    }
    try:
        if device in ("cuda", "mps"):
            free, total = torch.accelerator.get_memory_info()
            snap.update(
                total=int(total),
                free=int(free),
                allocated=int(torch.accelerator.memory_allocated()),
                peak=int(torch.accelerator.max_memory_allocated()),
            )
            if device == "cuda":
                snap["reserved"] = int(torch.cuda.memory_reserved())
            else:
                snap["reserved"] = int(torch.mps.driver_allocated_memory())
        else:
            snap.update(total=vm.total, free=vm.available, allocated=snap["process_rss"], peak=snap["process_rss"],
                        reserved=snap["process_rss"])
    except Exception as exc:  # telemetry must never take the worker down
        snap["error"] = str(exc)
    return snap


def memory_budget(device: str, override_gb: float | None = None, headroom: float = DEFAULT_HEADROOM) -> MemoryBudget:
    env_override = os.environ.get("NOELLE_MEMORY_BUDGET_GB")
    if env_override:
        override_gb = float(env_override)
    system_ram = psutil.virtual_memory().total
    if override_gb:
        return MemoryBudget(int(override_gb * GB), headroom, system_ram, device)
    if device in ("cuda", "mps"):
        try:
            _, total = torch.accelerator.get_memory_info()
            return MemoryBudget(int(total), headroom, system_ram, device)
        except Exception:
            pass
    return MemoryBudget(int(psutil.virtual_memory().available), headroom, system_ram, device)


def free_bytes(device: str) -> int:
    if device in ("cuda", "mps"):
        try:
            free, _ = torch.accelerator.get_memory_info()
            return int(free)
        except Exception:
            pass
    return int(psutil.virtual_memory().available)


def reset_peak(device: str) -> None:
    if device in ("cuda", "mps"):
        try:
            torch.accelerator.reset_peak_memory_stats()
        except Exception:
            pass


def peak_bytes(device: str) -> int | None:
    if device in ("cuda", "mps"):
        try:
            return int(torch.accelerator.max_memory_allocated())
        except Exception:
            return None
    return psutil.Process().memory_info().rss


def synchronize(device: str) -> None:
    try:
        if device == "cuda":
            torch.cuda.synchronize()
        elif device == "mps":
            torch.mps.synchronize()
    except Exception:
        pass


def empty_cache(device: str | None = None) -> None:
    """Return freed blocks to the driver: ``gc.collect()`` then the backend's cache flush."""
    gc.collect()
    try:
        if torch.cuda.is_available() and device in (None, "cuda"):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        if torch.backends.mps.is_available() and device in (None, "mps"):
            torch.mps.empty_cache()
    except Exception:
        pass


def is_oom(exc: BaseException) -> bool:
    if isinstance(exc, torch.OutOfMemoryError):
        return True
    text = str(exc).lower()
    return "out of memory" in text or "mps backend out of memory" in text or "cuda error: out of memory" in text


def is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def runtime_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
        "apple_silicon": is_apple_silicon(),
        "python": platform.python_version(),
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info["cuda"] = {
            "name": props.name,
            "total_memory": props.total_memory,
            "capability": f"{props.major}.{props.minor}",
            "multi_processor_count": props.multi_processor_count,
        }
    return info
