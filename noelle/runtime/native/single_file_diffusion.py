"""Single-file diffusion checkpoints (``.safetensors`` without ``model_index.json``)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from noelle.runtime import devices
from noelle.runtime.errors import DependencyMissing, LoadFailed
from noelle.runtime.pipelines import DiffusionEngine


def _pick_checkpoint(source: str) -> Path:
    root = Path(source).expanduser()
    if root.is_file():
        return root
    candidates = [path for path in root.rglob("*.safetensors") if ".fp16." not in path.name]
    if not candidates:
        candidates = list(root.rglob("*.safetensors")) or list(root.rglob("*.ckpt"))
    if not candidates:
        raise LoadFailed("No single-file diffusion checkpoint was found in this repository.")
    return max(candidates, key=lambda path: path.stat().st_size)


class SingleFileDiffusionEngine(DiffusionEngine):
    engine = "diffusers-single-file"

    @classmethod
    def load(cls, source: str, *, device: str, precision: str = "auto", revision: str | None = None,
             trust_remote_code: bool = False, **kwargs: Any) -> SingleFileDiffusionEngine:
        try:
            from diffusers import AutoPipelineForText2Image
        except ImportError as exc:
            raise DependencyMissing("Single-file diffusion needs the 'diffusers' package.",
                                    hint='Install the diffusion component in Settings › Components.',
                                    component="diffusers") from exc
        if not hasattr(AutoPipelineForText2Image, "from_single_file"):
            raise LoadFailed("This diffusers version cannot load single-file checkpoints.",
                             hint="Update the runtime environment (Settings › Environments).")
        checkpoint = _pick_checkpoint(source)
        dtype = devices.torch_dtype(device, precision)
        try:
            pipe = AutoPipelineForText2Image.from_single_file(
                str(checkpoint), torch_dtype=dtype, trust_remote_code=trust_remote_code)
        except TypeError:
            pipe = AutoPipelineForText2Image.from_single_file(str(checkpoint), dtype=dtype)
        pipe = pipe.to(device)
        if device == "mps" and hasattr(pipe, "enable_attention_slicing"):
            pipe.enable_attention_slicing()
        if hasattr(pipe, "set_progress_bar_config"):
            pipe.set_progress_bar_config(disable=True)
        return cls(pipe, device)