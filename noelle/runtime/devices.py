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

from noelle.runtime.preflight import DEFAULT_HEADROOM, GB, MemoryBudget


def _torch() -> Any:
    import torch
    return torch


def resolve_device(preference: str | None = "auto") -> str:
    pref = (preference or "auto").lower()
    if pref == "cpu":
        return "cpu"
    try:
        torch = _torch()
        cuda = torch.cuda.is_available()
        mps = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
    except Exception:
        cuda = mps = False
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
        try:
            return _torch().cuda.get_device_name(0)
        except Exception:
            return "NVIDIA GPU"
    if device == "mps":
        try:
            out = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True,
                                 timeout=2).stdout.strip()
            return f"{out} GPU" if out else "Apple Silicon GPU"
        except (OSError, subprocess.SubprocessError):
            return "Apple Silicon GPU"
    return platform.processor() or platform.machine() or "CPU"


def torch_dtype(device: str, precision: str | None) -> Any:
    torch = _torch()
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


def _memory_info(device: str) -> tuple[int, int]:
    """``(free, total)`` accelerator memory using APIs available on torch 2.4 as well as newer ones.

    ``torch.accelerator`` only exists from torch 2.5, while Noelle supports 2.4, so the
    backend-specific functions are preferred and the unified API is only a fallback.
    """
    torch = _torch()
    if device == "cuda":
        mem_get_info = getattr(torch.cuda, "mem_get_info", None)
        if mem_get_info is not None:
            free, total = mem_get_info()
            return int(free), int(total)
    elif device == "mps":
        recommended = getattr(torch.mps, "recommended_max_memory", None)
        driver = getattr(torch.mps, "driver_allocated_memory", None)
        if recommended is not None and driver is not None:
            total = int(recommended())
            return max(total - int(driver()), 0), total
    accelerator = getattr(torch, "accelerator", None)
    if accelerator is not None:
        free, total = accelerator.get_memory_info()
        return int(free), int(total)
    raise RuntimeError(f"memory information is unavailable for device '{device}'")


def _allocator_stats(device: str) -> tuple[int, int]:
    """``(allocated, peak)`` bytes held by live tensors in this process."""
    torch = _torch()
    accelerator = getattr(torch, "accelerator", None)
    if accelerator is not None:
        return int(accelerator.memory_allocated()), int(accelerator.max_memory_allocated())
    if device == "cuda":
        return int(torch.cuda.memory_allocated()), int(torch.cuda.max_memory_allocated())
    return int(torch.mps.current_allocated_memory()), int(torch.mps.driver_allocated_memory())


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
            torch = _torch()
            free, total = _memory_info(device)
            allocated, peak = _allocator_stats(device)
            snap.update(total=total, free=free, allocated=allocated, peak=peak)
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
            _, total = _memory_info(device)
            return MemoryBudget(int(total), headroom, system_ram, device)
        except Exception:
            pass
    return MemoryBudget(int(psutil.virtual_memory().available), headroom, system_ram, device)


def free_bytes(device: str) -> int:
    if device in ("cuda", "mps"):
        try:
            free, _ = _memory_info(device)
            return int(free)
        except Exception:
            pass
    return int(psutil.virtual_memory().available)


def reset_peak(device: str) -> None:
    if device in ("cuda", "mps"):
        try:
            torch = _torch()
            accelerator = getattr(torch, "accelerator", None)
            if accelerator is not None:
                accelerator.reset_peak_memory_stats()
            elif device == "cuda":
                torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass


def peak_bytes(device: str) -> int | None:
    if device in ("cuda", "mps"):
        try:
            return _allocator_stats(device)[1]
        except Exception:
            return None
    return psutil.Process().memory_info().rss


def synchronize(device: str) -> None:
    try:
        torch = _torch()
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
        torch = _torch()
        if torch.cuda.is_available() and device in (None, "cuda"):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        if torch.backends.mps.is_available() and device in (None, "mps"):
            torch.mps.empty_cache()
    except Exception:
        pass


def is_oom(exc: BaseException) -> bool:
    try:
        torch = _torch()
    except ImportError:
        torch = None
    if torch is not None and isinstance(exc, torch.OutOfMemoryError):
        return True
    text = str(exc).lower()
    return "out of memory" in text or "mps backend out of memory" in text or "cuda error: out of memory" in text


def is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def runtime_info() -> dict[str, Any]:
    torch = _torch()
    mps_backend = getattr(torch.backends, "mps", None)
    info: dict[str, Any] = {
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": bool(mps_backend and mps_backend.is_available()),
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
