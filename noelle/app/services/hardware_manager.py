"""Hardware telemetry for the status bar and compatibility badges.

The desktop process must not import torch, so it probes hardware with psutil
plus platform tools (``ioreg`` on macOS, ``nvidia-smi`` on NVIDIA systems) and
merges in the torch-level numbers (allocated/peak memory, Metal working-set
budget) that the worker reports in its heartbeat.
"""

from __future__ import annotations

import platform
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any

import psutil
from PySide6.QtCore import QObject, QTimer, Signal

from noelle.runtime.preflight import DEFAULT_HEADROOM, GB, MemoryBudget

_IOREG_UTIL = re.compile(r'"Device Utilization %"=(\d+)')
_IOREG_INUSE = re.compile(r'"In use system memory"=(\d+)')
_IOREG_MODEL = re.compile(r'"model" = "([^"]+)"')


@dataclass
class GpuInfo:
    kind: str  # cuda | mps | cpu
    name: str
    total_memory: int | None = None  # dedicated VRAM, or recommended Metal working set
    cores: int | None = None


@dataclass
class HardwareMetrics:
    timestamp: float
    cpu_percent: float
    cpu_count: int
    ram_total: int
    ram_used: int
    ram_available: int
    gpu: GpuInfo
    gpu_util: float | None = None
    vram_used: int | None = None
    vram_total: int | None = None
    worker_allocated: int | None = None
    worker_peak: int | None = None
    budget_usable: int | None = None
    loaded_models: list[dict[str, Any]] = field(default_factory=list)
    runtime_state: str = "stopped"


def _run(args: list[str], timeout: float = 2.0) -> str:
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=timeout).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def detect_gpu() -> GpuInfo:
    if shutil.which("nvidia-smi"):
        out = _run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"])
        line = out.strip().splitlines()[0] if out.strip() else ""
        if line and "," in line:
            name, total = [part.strip() for part in line.split(",", 1)]
            try:
                return GpuInfo("cuda", name, int(float(total)) * 1024 * 1024)
            except ValueError:
                return GpuInfo("cuda", name)
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        out = _run(["ioreg", "-r", "-d", "1", "-w", "0", "-c", "IOAccelerator"])
        match = _IOREG_MODEL.search(out)
        cores = re.search(r'"gpu-core-count" = (\d+)', out)
        name = match.group(1) if match else "Apple Silicon"
        ram = psutil.virtual_memory().total
        # Metal's recommendedMaxWorkingSetSize is ~75% of unified memory on 16-32 GB machines;
        # the worker replaces this estimate with the real value once it is up.
        return GpuInfo("mps", f"{name} GPU", int(ram * 0.75), int(cores.group(1)) if cores else None)
    return GpuInfo("cpu", platform.processor() or platform.machine() or "CPU", None)


def sample_gpu(gpu: GpuInfo) -> tuple[float | None, int | None]:
    """Return (utilization %, memory in use) without torch."""
    if gpu.kind == "cuda":
        out = _run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"])
        if out.strip():
            try:
                util, used = [float(x.strip()) for x in out.strip().splitlines()[0].split(",")[:2]]
                return util, int(used) * 1024 * 1024
            except ValueError:
                return None, None
    if gpu.kind == "mps":
        out = _run(["ioreg", "-r", "-d", "1", "-w", "0", "-c", "IOAccelerator"])
        util = _IOREG_UTIL.search(out)
        used = _IOREG_INUSE.search(out)
        return (float(util.group(1)) if util else None, int(used.group(1)) if used else None)
    return None, None


class HardwareManager(QObject):
    """Polls CPU/RAM every second and the GPU every two seconds (off the GUI thread)."""

    hardware_telemetry_updated = Signal(object)

    def __init__(self, runtime: Any | None = None, parent: QObject | None = None, interval_ms: int = 1000) -> None:
        super().__init__(parent)
        self.runtime = runtime
        self.gpu = detect_gpu()
        self.cpu_count = psutil.cpu_count(logical=True) or 1
        self.latest: HardwareMetrics | None = None
        self._gpu_sample: tuple[float | None, int | None] = (None, None)
        self._tick = 0
        self._busy = False
        psutil.cpu_percent(None)  # prime the counter
        self._timer = QTimer(self)
        self._timer.setInterval(interval_ms)
        self._timer.timeout.connect(self._poll)
        if runtime is not None:
            runtime.heartbeat.connect(self._on_heartbeat)

    def start(self) -> None:
        self._poll()
        self._timer.start()

    def stop(self) -> None:
        self._timer.stop()

    # ------------------------------------------------------------- budget
    def memory_budget(self, headroom: float = DEFAULT_HEADROOM) -> MemoryBudget:
        ram = psutil.virtual_memory().total
        beat = getattr(self.runtime, "last_heartbeat_payload", None) or {}
        budget = beat.get("budget") or {}
        if budget.get("total"):
            return MemoryBudget(int(budget["total"]), headroom, ram, beat.get("device") or self.gpu.kind)
        if self.gpu.total_memory:
            return MemoryBudget(int(self.gpu.total_memory), headroom, ram, self.gpu.kind)
        return MemoryBudget(int(ram * 0.7), headroom, ram, "cpu")

    # ------------------------------------------------------------- polling
    def _poll(self) -> None:
        if self._busy:
            return
        self._busy = True
        sample_gpu_now = self._tick % 2 == 0
        self._tick += 1
        from noelle.app.services.task_runner import runner

        runner().submit(self._sample, sample_gpu_now, on_result=self._publish,
                        on_finished=lambda: setattr(self, "_busy", False))

    def _sample(self, sample_gpu_now: bool) -> dict[str, Any]:
        vm = psutil.virtual_memory()
        data: dict[str, Any] = {
            "cpu_percent": psutil.cpu_percent(None),
            "ram_total": vm.total,
            "ram_used": vm.total - vm.available,
            "ram_available": vm.available,
        }
        if sample_gpu_now:
            data["gpu_sample"] = sample_gpu(self.gpu)
        return data

    def _publish(self, data: dict[str, Any]) -> None:
        if "gpu_sample" in data:
            self._gpu_sample = data["gpu_sample"]
        beat = getattr(self.runtime, "last_heartbeat_payload", None) or {}
        memory = beat.get("memory") or {}
        budget = beat.get("budget") or {}
        util, used = self._gpu_sample
        vram_total = memory.get("total") or self.gpu.total_memory
        metrics = HardwareMetrics(
            timestamp=time.time(),
            cpu_percent=float(data["cpu_percent"]),
            cpu_count=self.cpu_count,
            ram_total=int(data["ram_total"]),
            ram_used=int(data["ram_used"]),
            ram_available=int(data["ram_available"]),
            gpu=self.gpu,
            gpu_util=util,
            vram_used=used if used is not None else memory.get("allocated"),
            vram_total=vram_total,
            worker_allocated=memory.get("allocated"),
            worker_peak=memory.get("peak"),
            budget_usable=budget.get("usable"),
            loaded_models=list(beat.get("loaded") or []),
            runtime_state=getattr(self.runtime, "state", "stopped"),
        )
        self.latest = metrics
        self.hardware_telemetry_updated.emit(metrics)

    def _on_heartbeat(self, payload: dict[str, Any]) -> None:
        memory = payload.get("memory") or {}
        if payload.get("device") == "mps" and memory.get("total"):
            self.gpu.total_memory = int(memory["total"])
        if payload.get("device") == "cuda" and memory.get("total"):
            self.gpu.total_memory = int(memory["total"])


def probe_profile(headroom: float = DEFAULT_HEADROOM, backends: dict[str, bool] | None = None) -> Any:
    """Hardware profile for the advisor without a running worker (CLI, MCP server, tests).

    Uses the same fallbacks as :meth:`HardwareManager.memory_budget` when no worker heartbeat exists.
    """
    from noelle.runtime import advisor, capabilities

    gpu = detect_gpu()
    ram = psutil.virtual_memory().total
    if gpu.total_memory:
        budget = MemoryBudget(int(gpu.total_memory), headroom, ram, gpu.kind)
    else:
        budget = MemoryBudget(int(ram * 0.7), headroom, ram, "cpu")
    return advisor.HardwareProfile(budget.device, gpu.name, budget,
                                   dict(backends if backends is not None else capabilities.detect_backends_cached()),
                                   advisor.memory_bandwidth(gpu.name, budget.device))


def describe_gb(value: int | None) -> str:
    return "—" if value is None else f"{value / GB:.1f} GB"
