"""Audio, vision-pipeline, diffusion and embedding engines."""

from __future__ import annotations

import base64
import inspect
import io
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable

import numpy as np

from noelle import paths
from noelle.runtime import audio_dsp, devices
from noelle.runtime.errors import BadRequest, Cancelled, require


def _output_path(suffix: str, stem: str = "output") -> Path:
    out_dir = paths.outputs_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{stem}-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}{suffix}"


def _common(trust_remote_code: bool, revision: str | None) -> dict[str, Any]:
    common: dict[str, Any] = {"trust_remote_code": trust_remote_code}
    if revision:
        common["revision"] = revision
    return common


# ---------------------------------------------------------------------------
# Speech recognition
# ---------------------------------------------------------------------------


class AsrEngine:
    engine = "transformers-asr"

    def __init__(self, pipe: Any, model_type: str, device: str) -> None:
        self.pipe = pipe
        self.model_type = model_type
        self.device = device

    @classmethod
    def load(cls, source: str, *, device: str, precision: str = "auto", trust_remote_code: bool = False,
             revision: str | None = None) -> AsrEngine:
        from transformers import pipeline

        pipe = pipeline("automatic-speech-recognition", model=source, device=device,
                        dtype=devices.torch_dtype(device, precision), **_common(trust_remote_code, revision))
        return cls(pipe, getattr(pipe.model.config, "model_type", ""), device)

    def transcribe(self, audio_path: str, *, language: str | None = None, translate: bool = False,
                   timestamps: bool = True) -> dict[str, Any]:
        sampling_rate = int(getattr(self.pipe.feature_extractor, "sampling_rate", 16000))
        audio, sr = audio_dsp.load_audio(audio_path, sampling_rate)
        duration = len(audio) / sr if sr else 0.0
        kwargs: dict[str, Any] = {}
        generate_kwargs: dict[str, Any] = {}
        is_whisper = "whisper" in self.model_type
        if timestamps:
            kwargs["return_timestamps"] = True if is_whisper else "word"
        if is_whisper:
            if language:
                generate_kwargs["language"] = language
            if translate:
                generate_kwargs["task"] = "translate"
            if duration > 30:
                kwargs["chunk_length_s"] = 30
        if generate_kwargs:
            kwargs["generate_kwargs"] = generate_kwargs
        t0 = time.perf_counter()
        out = self.pipe({"raw": audio, "sampling_rate": sr}, **kwargs)
        elapsed = time.perf_counter() - t0
        chunks = []
        for chunk in out.get("chunks", []) or []:
            start, end = (chunk.get("timestamp") or (None, None))
            chunks.append({"start": start, "end": end, "text": str(chunk.get("text", "")).strip()})
        return {
            "text": str(out.get("text", "")).strip(),
            "chunks": chunks,
            "duration_s": duration,
            "metrics": {"total_ms": elapsed * 1000.0, "real_time_factor": (elapsed / duration) if duration else None},
            "envelope": audio_dsp.waveform_envelope(audio, 600),
        }

    def release(self) -> None:
        self.pipe = None


# ---------------------------------------------------------------------------
# Speech synthesis
# ---------------------------------------------------------------------------


class TtsEngine:
    engine = "transformers-tts"

    def __init__(self, pipe: Any, model_type: str) -> None:
        self.pipe = pipe
        self.model_type = model_type

    @classmethod
    def load(cls, source: str, *, device: str, precision: str = "auto", trust_remote_code: bool = False,
             revision: str | None = None) -> TtsEngine:
        from transformers import pipeline

        pipe = pipeline("text-to-audio", model=source, device=device, **_common(trust_remote_code, revision))
        return cls(pipe, getattr(pipe.model.config, "model_type", ""))

    def synthesize(self, text: str, *, voice: int = 0, speed: float = 1.0, pitch: float = 0.0) -> dict[str, Any]:
        import soundfile as sf

        if not text.strip():
            raise BadRequest("Nothing to synthesize: the text is empty.")
        forward_params: dict[str, Any] = {}
        if self.model_type == "speecht5":
            import torch

            generator = torch.Generator().manual_seed(int(voice))
            forward_params["speaker_embeddings"] = torch.randn((1, 512), generator=generator) * 0.05
        t0 = time.perf_counter()
        out = self.pipe(text, forward_params=forward_params) if forward_params else self.pipe(text)
        elapsed = time.perf_counter() - t0
        audio = np.asarray(out["audio"], dtype=np.float32)
        sr = int(out["sampling_rate"])
        if audio.ndim > 1:
            audio = audio.squeeze()
            if audio.ndim > 1:  # (channels, samples) -> mono
                audio = audio.mean(axis=0)
        if speed and abs(speed - 1.0) > 1e-3:
            audio = audio_dsp.change_speed(audio, sr, speed)
        if pitch and abs(pitch) > 1e-3:
            audio = audio_dsp.pitch_shift(audio, sr, pitch)
        audio = audio_dsp.peak_normalize(audio)
        path = _output_path(".wav", "speech")
        sf.write(str(path), audio, sr)
        duration = len(audio) / sr if sr else 0.0
        return {
            "audio_path": str(path),
            "sampling_rate": sr,
            "duration_s": duration,
            "metrics": {"total_ms": elapsed * 1000.0, "real_time_factor": (elapsed / duration) if duration else None},
            "envelope": audio_dsp.waveform_envelope(audio, 600),
        }

    def release(self) -> None:
        self.pipe = None


# ---------------------------------------------------------------------------
# Detection / classification
# ---------------------------------------------------------------------------


class VisionPipelineEngine:
    engine = "transformers-vision-pipeline"

    def __init__(self, pipe: Any, task: str) -> None:
        self.pipe = pipe
        self.task = task

    @classmethod
    def load(cls, source: str, *, task: str, device: str, trust_remote_code: bool = False,
             revision: str | None = None) -> VisionPipelineEngine:
        from transformers import pipeline

        pipe = pipeline(task, model=source, device=device, **_common(trust_remote_code, revision))
        return cls(pipe, task)

    def run(self, image_path: str, *, threshold: float = 0.5, top_k: int = 5) -> dict[str, Any]:
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        t0 = time.perf_counter()
        if self.task == "object-detection":
            raw = self.pipe(image, threshold=threshold)
            detections = [{"label": d["label"], "score": float(d["score"]), "box": {k: float(v) for k, v in d["box"].items()}}
                          for d in raw]
            result: dict[str, Any] = {"detections": detections}
        else:
            raw = self.pipe(image, top_k=top_k)
            result = {"labels": [{"label": d["label"], "score": float(d["score"])} for d in raw]}
        result["image_size"] = {"width": image.width, "height": image.height}
        result["metrics"] = {"total_ms": (time.perf_counter() - t0) * 1000.0}
        return result

    def release(self) -> None:
        self.pipe = None


# ---------------------------------------------------------------------------
# Diffusion
# ---------------------------------------------------------------------------

# Approximate linear map from SD 1.x/2.x latents to RGB, used for live previews.
_SD_LATENT_RGB = np.array(
    [[0.3512, 0.2297, 0.3227], [0.3250, 0.4974, 0.2350], [-0.2829, 0.1762, 0.2721], [-0.2120, -0.2616, -0.7177]],
    dtype=np.float32,
)


def latents_preview_png(latents: Any, size: int = 192) -> bytes | None:
    """Cheap RGB preview of the current latent tensor (no VAE decode)."""
    try:
        from PIL import Image

        lat = latents[0].detach().float().cpu().numpy()  # (C, H, W)
        if lat.ndim != 3:
            return None
        if lat.shape[0] == 4:
            rgb = np.einsum("chw,cr->hwr", lat, _SD_LATENT_RGB)
            rgb = (rgb + 1.0) * 0.5
        else:
            rgb = np.moveaxis(lat[:3], 0, -1)
            lo, hi = float(rgb.min()), float(rgb.max())
            rgb = (rgb - lo) / (hi - lo + 1e-6)
        image = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)).resize((size, size))
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return buf.getvalue()
    except Exception:
        return None


class DiffusionEngine:
    engine = "diffusers"

    def __init__(self, pipe: Any, device: str, img2img: Any = None) -> None:
        self.pipe = pipe
        self.device = device
        self._img2img = img2img

    @classmethod
    def load(cls, source: str, *, device: str, precision: str = "auto", revision: str | None = None,
             trust_remote_code: bool = False) -> DiffusionEngine:
        require("diffusers", "diffusion")
        from diffusers import AutoPipelineForText2Image, DiffusionPipeline

        dtype = devices.torch_dtype(device, precision)
        # newer diffusers renamed torch_dtype to dtype (the old name still works but warns)
        kwargs: dict[str, Any] = {"dtype" if _accepts_dtype_kwarg() else "torch_dtype": dtype}
        if revision:
            kwargs["revision"] = revision
        try:
            pipe = AutoPipelineForText2Image.from_pretrained(source, **kwargs)
        except Exception:
            pipe = DiffusionPipeline.from_pretrained(source, trust_remote_code=trust_remote_code, **kwargs)
        pipe = pipe.to(device)
        if device == "mps" and hasattr(pipe, "enable_attention_slicing"):
            pipe.enable_attention_slicing()
        if hasattr(pipe, "set_progress_bar_config"):
            pipe.set_progress_bar_config(disable=True)
        return cls(pipe, device)

    def _image_to_image(self) -> Any:
        if self._img2img is None:
            from diffusers import AutoPipelineForImage2Image

            self._img2img = AutoPipelineForImage2Image.from_pipe(self.pipe)
        return self._img2img

    def generate(self, *, prompt: str, negative_prompt: str | None = None, steps: int = 25, guidance: float = 7.0,
                 seed: int | None = None, width: int = 512, height: int = 512, init_image: str | None = None,
                 strength: float = 0.6, stop_event: threading.Event | None = None,
                 on_step: Callable[[int, int, bytes | None], None] | None = None,
                 preview_every: int = 1) -> dict[str, Any]:
        import torch

        if not prompt.strip():
            raise BadRequest("The prompt is empty.")
        seed = int(seed) if seed is not None and int(seed) >= 0 else int(torch.randint(0, 2**31 - 1, (1,)).item())
        generator = torch.Generator(device="cpu").manual_seed(seed)
        width = max(64, int(width) // 8 * 8)
        height = max(64, int(height) // 8 * 8)
        steps = max(1, int(steps))
        pipe = self._image_to_image() if init_image else self.pipe

        def on_step_end(pipeline: Any, step: int, timestep: Any, callback_kwargs: dict[str, Any]) -> dict[str, Any]:
            if stop_event is not None and stop_event.is_set():
                pipeline._interrupt = True
            if on_step is not None:
                preview = None
                if preview_every and (step % preview_every == 0) and "latents" in callback_kwargs:
                    preview = latents_preview_png(callback_kwargs["latents"])
                on_step(step + 1, steps, preview)
            return callback_kwargs

        call: dict[str, Any] = {"prompt": prompt, "num_inference_steps": steps, "guidance_scale": float(guidance),
                                "generator": generator}
        params = inspect.signature(pipe.__call__).parameters
        if negative_prompt and "negative_prompt" in params:
            call["negative_prompt"] = negative_prompt
        if init_image:
            from PIL import Image

            call["image"] = Image.open(init_image).convert("RGB").resize((width, height))
            call["strength"] = float(strength)
        else:
            call.update(width=width, height=height)
        if "callback_on_step_end" in params:
            call["callback_on_step_end"] = on_step_end
            if "callback_on_step_end_tensor_inputs" in params:
                call["callback_on_step_end_tensor_inputs"] = ["latents"]
        devices.reset_peak(self.device)
        t0 = time.perf_counter()
        result = pipe(**call)
        elapsed = time.perf_counter() - t0
        if stop_event is not None and stop_event.is_set():
            raise Cancelled("Image generation was stopped.")
        image = result.images[0]
        path = _output_path(".png", "image")
        image.save(path)
        return {
            "image_path": str(path),
            "seed": seed,
            "width": image.width,
            "height": image.height,
            "metrics": {"total_ms": elapsed * 1000.0, "seconds_per_step": elapsed / steps,
                        "peak_memory_bytes": devices.peak_bytes(self.device)},
        }

    def release(self) -> None:
        self.pipe = None
        self._img2img = None


def _accepts_dtype_kwarg() -> bool:
    """True when DiffusionPipeline.from_pretrained understands ``dtype=`` (checked in its source, not by version)."""
    try:
        from diffusers import DiffusionPipeline

        return 'pop("dtype"' in inspect.getsource(DiffusionPipeline.from_pretrained)
    except Exception:
        return False


def encode_png_b64(data: bytes | None) -> str | None:
    return base64.b64encode(data).decode("ascii") if data else None


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------


class EmbeddingEngine:
    engine = "sentence-transformers"

    def __init__(self, model: Any, device: str) -> None:
        self.model = model
        self.device = device

    @classmethod
    def load(cls, source: str, *, device: str, trust_remote_code: bool = False,
             revision: str | None = None) -> EmbeddingEngine:
        require("sentence_transformers", "rag")
        from sentence_transformers import SentenceTransformer

        kwargs: dict[str, Any] = {"device": device, "trust_remote_code": trust_remote_code}
        if revision:
            kwargs["revision"] = revision
        return cls(SentenceTransformer(source, **kwargs), device)

    @property
    def tokenizer(self) -> Any:
        return self.model.tokenizer

    @property
    def dimension(self) -> int:
        dim = None
        for name in ("get_embedding_dimension", "get_sentence_embedding_dimension"):
            fn = getattr(self.model, name, None)
            if fn is not None:
                dim = fn()
                break
        if not dim:
            dim = int(self.encode(["dimension probe"]).shape[1])
        return int(dim)

    @property
    def max_seq_length(self) -> int:
        return int(getattr(self.model, "max_seq_length", 512) or 512)

    def encode(self, texts: list[str], *, batch_size: int = 32, normalize: bool = True) -> np.ndarray:
        vectors = self.model.encode(list(texts), batch_size=batch_size, normalize_embeddings=normalize,
                                    convert_to_numpy=True, show_progress_bar=False)
        return np.asarray(vectors, dtype=np.float32)

    def release(self) -> None:
        self.model = None
