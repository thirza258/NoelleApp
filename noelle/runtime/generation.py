"""Text and vision-language engines with token streaming.

Three text backends share one interface (``stream_chat``):

* :class:`TransformersTextEngine` - PyTorch on CUDA, Apple Metal (MPS) or CPU,
  including bitsandbytes / AWQ / GPTQ checkpoints where the hardware allows.
* :class:`MlxTextEngine` - Apple Silicon 4-bit checkpoints produced by
  ``mlx_lm.convert`` (the Mac path for in-app quantization).
* :class:`LlamaCppTextEngine` - GGUF files through llama-cpp-python.

Every engine reports the same metrics: time-to-first-token, tokens/second,
prompt/completion token counts and peak accelerator memory.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from noelle.runtime import devices
from noelle.runtime.errors import LoadFailed, Unsupported, require
from noelle.runtime.preflight import context_limit, kv_cache_bytes_per_token, tune_generation
from noelle.runtime.protocol import GenerationParams

OnText = Callable[[str], None]
MIN_NEW_TOKENS = 16
_MAX_SANE_CONTEXT = 10_000_000


@dataclass
class GenerationResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    ttft_ms: float | None
    tokens_per_second: float | None
    total_ms: float
    peak_memory_bytes: int | None
    finish_reason: str  # stop | length | cancelled
    tuning: dict[str, Any] | None = None

    def metrics(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("text")
        return data


def _rates(t0: float, first: float | None, end: float, completion: int) -> tuple[float | None, float | None]:
    ttft = (first - t0) * 1000.0 if first is not None else None
    if completion <= 0:
        return ttft, None
    if first is not None and completion > 1 and end > first:
        return ttft, (completion - 1) / (end - first)
    return ttft, completion / max(end - t0, 1e-6)


def messages_with_system(messages: list[dict[str, Any]], system_prompt: str | None) -> list[dict[str, Any]]:
    msgs = [{"role": str(m.get("role", "user")), "content": _content_text(m.get("content"))}
            for m in messages if m.get("content") is not None]
    if system_prompt:
        if msgs and msgs[0]["role"] == "system":
            msgs[0] = {"role": "system", "content": f"{system_prompt}\n\n{msgs[0]['content']}"}
        else:
            msgs.insert(0, {"role": "system", "content": system_prompt})
    return msgs


def _content_text(content: Any) -> str:
    # OpenAI-style content arrays: keep the text parts.
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return str(content)


def merge_system_into_user(msgs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """For chat templates that reject a system role."""
    system = "\n\n".join(m["content"] for m in msgs if m["role"] == "system")
    rest = [dict(m) for m in msgs if m["role"] != "system"]
    if system:
        if rest and rest[0]["role"] == "user":
            rest[0]["content"] = f"{system}\n\n{rest[0]['content']}"
        else:
            rest.insert(0, {"role": "user", "content": system})
    return rest


def render_plain_prompt(msgs: list[dict[str, Any]]) -> str:
    lines = [f"{m['role'].capitalize()}: {m['content']}" for m in msgs]
    lines.append("Assistant:")
    return "\n\n".join(lines)


def drop_oldest_turn(msgs: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
    """Remove the oldest non-system message, keeping at least the final user turn."""
    non_system = [i for i, m in enumerate(msgs) if m["role"] != "system"]
    if len(non_system) <= 1:
        return None
    victim = non_system[0]
    return msgs[:victim] + msgs[victim + 1:]


def _apply_stop_strings(text: str, stops: list[str]) -> tuple[str, bool]:
    for stop in stops:
        index = text.find(stop)
        if index != -1:
            return text[:index], True
    return text, False


# ---------------------------------------------------------------------------
# transformers
# ---------------------------------------------------------------------------


def _streamer_class():
    from transformers.generation.streamers import TextStreamer

    class EmitStreamer(TextStreamer):
        """TextStreamer that forwards decoded text instead of printing it."""

        def __init__(self, tokenizer: Any, on_text: OnText, **kwargs: Any) -> None:
            super().__init__(tokenizer, **kwargs)
            self._on_text = on_text
            self.first_token_at: float | None = None
            self.generated = 0
            self.parts: list[str] = []

        def put(self, value: Any) -> None:
            if self.skip_prompt and self.next_tokens_are_prompt:
                self.next_tokens_are_prompt = False
                return
            if self.first_token_at is None:
                self.first_token_at = time.perf_counter()
            self.generated += int(value.numel())
            super().put(value)

        def on_finalized_text(self, text: str, stream_end: bool = False) -> None:
            if text:
                self.parts.append(text)
                self._on_text(text)

    return EmitStreamer


def _event_stop(event: threading.Event):
    import torch
    from transformers import StoppingCriteria

    class EventStop(StoppingCriteria):
        def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> Any:
            return torch.full((input_ids.shape[0],), event.is_set(), dtype=torch.bool, device=input_ids.device)

    return EventStop()


def _sampling_kwargs(params: GenerationParams, pad_token_id: int | None) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"repetition_penalty": params.repetition_penalty}
    if pad_token_id is not None:
        kwargs["pad_token_id"] = pad_token_id
    if params.temperature > 0:
        kwargs.update(do_sample=True, temperature=params.temperature, top_p=params.top_p,
                      top_k=params.top_k if params.top_k > 0 else None)
    else:
        kwargs.update(do_sample=False, temperature=None, top_p=None, top_k=None)
    return kwargs


def _model_kwargs_for(device: str, precision: str) -> dict[str, Any]:
    import torch

    if precision in ("int8", "nf4"):
        if device != "cuda":
            raise Unsupported(f"{precision} (bitsandbytes) needs an NVIDIA CUDA GPU; this runtime is on {device}.",
                              hint="On Apple Silicon use 'Quantize to 4-bit MLX' instead.")
        require("bitsandbytes", "cuda-quant")
        from transformers import BitsAndBytesConfig

        if precision == "int8":
            quant = BitsAndBytesConfig(load_in_8bit=True)
        else:
            quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                       bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
        return {"quantization_config": quant, "device_map": "auto"}
    if precision in ("awq-4bit", "gptq-4bit") and device != "cuda":
        raise Unsupported(f"{precision} kernels need an NVIDIA CUDA GPU; this runtime is on {device}.",
                          hint="Pick an FP16 or MLX 4-bit build of this model instead.")
    kwargs: dict[str, Any] = {"dtype": devices.torch_dtype(device, precision)}
    if device != "cpu":
        kwargs["device_map"] = device
    return kwargs


class TransformersTextEngine:
    engine = "transformers"

    def __init__(self, model: Any, tokenizer: Any, device: str, config: dict[str, Any],
                 speculative: Any = None, assistant_model: Any = None) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.config = config
        self.speculative = speculative
        self.assistant_model = assistant_model
        window = context_limit(config)
        if not window:
            tok_max = getattr(tokenizer, "model_max_length", None)
            window = tok_max if isinstance(tok_max, int) and tok_max < _MAX_SANE_CONTEXT else None
        self.context_window = window
        self.kv_per_token = kv_cache_bytes_per_token(config)

    @classmethod
    def load(cls, source: str, *, device: str, precision: str = "auto", trust_remote_code: bool = False,
             revision: str | None = None, gguf_file: str | None = None,
             speculative: dict[str, Any] | None = None) -> TransformersTextEngine:
        from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

        common: dict[str, Any] = {"trust_remote_code": trust_remote_code}
        if revision:
            common["revision"] = revision
        if gguf_file:
            require("gguf", "gguf")
            common["gguf_file"] = gguf_file
        config = AutoConfig.from_pretrained(source, **common)
        tokenizer = AutoTokenizer.from_pretrained(source, **common)
        model_cls = AutoModelForSeq2SeqLM if getattr(config, "is_encoder_decoder", False) else AutoModelForCausalLM
        model = model_cls.from_pretrained(source, **common, **_model_kwargs_for(device, precision))
        model.eval()
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token

        from noelle.runtime.speculative import SpeculativeConfig, compatibility

        spec = SpeculativeConfig.from_mapping(speculative)
        assistant = None
        if spec.mode == "draft" and not spec.draft_model:
            raise Unsupported("Speculative decoding needs a draft model.",
                              hint="Choose a compatible draft model, or use prompt lookup.")
        if spec.mode == "draft" and spec.draft_model:
            draft_config = AutoConfig.from_pretrained(spec.draft_model, trust_remote_code=trust_remote_code)
            draft_tokenizer = AutoTokenizer.from_pretrained(spec.draft_model, trust_remote_code=trust_remote_code)
            ok, reason = compatibility(config.to_dict(), draft_config.to_dict(),
                                       target_tokenizer=tokenizer, draft_tokenizer=draft_tokenizer)
            if not ok:
                raise Unsupported(f"Speculative decoding is unavailable: {reason}",
                                  hint="Choose a draft model with the same tokenizer, or use prompt lookup.")
            assistant = AutoModelForCausalLM.from_pretrained(
                spec.draft_model, trust_remote_code=trust_remote_code, **_model_kwargs_for(device, precision))
            assistant.eval()
        return cls(model, tokenizer, device, config.to_dict(), speculative=spec, assistant_model=assistant)

    # -- prompt handling -------------------------------------------------
    def encode_messages(self, msgs: list[dict[str, Any]]) -> dict[str, Any]:
        tok = self.tokenizer
        if getattr(tok, "chat_template", None):
            try:
                enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)
            except Exception:
                enc = tok.apply_chat_template(merge_system_into_user(msgs), add_generation_prompt=True,
                                              return_tensors="pt", return_dict=True)
        else:
            enc = tok(render_plain_prompt(msgs), return_tensors="pt")
        device = self.model.device
        return {k: enc[k].to(device) for k in ("input_ids", "attention_mask") if k in enc}

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def _fit(self, msgs: list[dict[str, Any]]) -> tuple[dict[str, Any], int]:
        enc = self.encode_messages(msgs)
        n = int(enc["input_ids"].shape[-1])
        if not self.context_window:
            return enc, n
        limit = self.context_window - MIN_NEW_TOKENS
        while n > limit:
            shorter = drop_oldest_turn(msgs)
            if shorter is None:
                break
            msgs = shorter
            enc = self.encode_messages(msgs)
            n = int(enc["input_ids"].shape[-1])
        if n > limit:  # a single huge turn: keep its tail
            enc = {k: v[:, -limit:] for k, v in enc.items()}
            n = limit
        return enc, n

    # -- generation ------------------------------------------------------
    def stream_chat(self, messages: list[dict[str, Any]], params: GenerationParams, *, stop_event: threading.Event,
                    on_text: OnText, system_prompt: str | None = None) -> GenerationResult:
        msgs = messages_with_system(messages, system_prompt)
        enc, prompt_tokens = self._fit(msgs)
        tuned = tune_generation(params.max_new_tokens, prompt_tokens, context_window=self.context_window,
                                kv_bytes_per_token=self.kv_per_token,
                                free_bytes=devices.free_bytes(self.device) if self.device != "cpu" else None)
        return self._generate(enc, prompt_tokens, params, tuned, stop_event, on_text)

    def _generate(self, enc: dict[str, Any], prompt_tokens: int, params: GenerationParams, tuned: Any,
                  stop_event: threading.Event, on_text: OnText, tokenizer: Any = None) -> GenerationResult:
        import torch
        from transformers import StoppingCriteriaList

        tokenizer = tokenizer or self.tokenizer
        streamer = _streamer_class()(tokenizer, on_text, skip_prompt=True, skip_special_tokens=True)
        kwargs = _sampling_kwargs(params, getattr(tokenizer, "pad_token_id", None))
        kwargs.update(enc)
        kwargs.update(max_new_tokens=tuned.max_new_tokens, streamer=streamer,
                      stopping_criteria=StoppingCriteriaList([_event_stop(stop_event)]))
        if params.stop:
            kwargs.update(stop_strings=params.stop, tokenizer=tokenizer)
        if self.assistant_model is not None:
            kwargs["assistant_model"] = self.assistant_model
        elif self.speculative is not None and getattr(self.speculative, "mode", "off") == "prompt_lookup":
            kwargs["prompt_lookup_num_tokens"] = int(getattr(self.speculative, "draft_tokens", 10))
        if params.seed is not None:
            torch.manual_seed(params.seed)
        devices.reset_peak(self.device)
        t0 = time.perf_counter()
        with torch.inference_mode():
            self.model.generate(**kwargs)
        devices.synchronize(self.device)
        end = time.perf_counter()
        ttft, rate = _rates(t0, streamer.first_token_at, end, streamer.generated)
        text = "".join(streamer.parts)
        if params.stop:
            text, _ = _apply_stop_strings(text, params.stop)
        if stop_event.is_set():
            finish = "cancelled"
        elif streamer.generated >= tuned.max_new_tokens:
            finish = "length"
        else:
            finish = "stop"
        return GenerationResult(
            text=text, prompt_tokens=prompt_tokens, completion_tokens=streamer.generated, ttft_ms=ttft,
            tokens_per_second=rate, total_ms=(end - t0) * 1000.0, peak_memory_bytes=devices.peak_bytes(self.device),
            finish_reason=finish,
            tuning={"max_new_tokens": tuned.max_new_tokens, "clamped": tuned.clamped, "reason": tuned.reason},
        )

    def complete(self, prompt: str, *, max_new_tokens: int = 96, system_prompt: str | None = None) -> str:
        params = GenerationParams(temperature=0.0, max_new_tokens=max_new_tokens, repetition_penalty=1.05)
        result = self.stream_chat([{"role": "user", "content": prompt}], params, stop_event=threading.Event(),
                                  on_text=lambda _t: None, system_prompt=system_prompt)
        return result.text

    def release(self) -> None:
        self.model = None
        self.tokenizer = None
        self.assistant_model = None


# ---------------------------------------------------------------------------
# MLX (Apple Silicon)
# ---------------------------------------------------------------------------


def is_mlx_checkpoint(path: str | Path | None) -> bool:
    """MLX conversions record their quantization in config.json under "quantization"."""
    if not path:
        return False
    config = Path(path) / "config.json"
    if not config.is_file():
        return False
    try:
        import json

        data = json.loads(config.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    quant = data.get("quantization")
    return isinstance(quant, dict) and "bits" in quant and "group_size" in quant


class MlxTextEngine:
    engine = "mlx"

    def __init__(self, model: Any, tokenizer: Any, config: dict[str, Any]) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = "mps"
        self.context_window = context_limit(config)
        self.kv_per_token = kv_cache_bytes_per_token(config)

    @classmethod
    def load(cls, source: str, *, trust_remote_code: bool = False) -> MlxTextEngine:
        if not devices.is_apple_silicon():
            raise Unsupported("MLX checkpoints need an Apple Silicon Mac.")
        require("mlx_lm", "mlx")
        from mlx_lm import load

        model, tokenizer, config = load(source, return_config=True,
                                        tokenizer_config={"trust_remote_code": trust_remote_code})
        return cls(model, tokenizer, dict(config))

    def _encode(self, msgs: list[dict[str, Any]]) -> list[int]:
        tok = self.tokenizer
        if getattr(tok, "chat_template", None):
            try:
                return list(tok.apply_chat_template(msgs, add_generation_prompt=True))
            except Exception:
                return list(tok.apply_chat_template(merge_system_into_user(msgs), add_generation_prompt=True))
        return list(tok.encode(render_plain_prompt(msgs)))

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def stream_chat(self, messages: list[dict[str, Any]], params: GenerationParams, *, stop_event: threading.Event,
                    on_text: OnText, system_prompt: str | None = None) -> GenerationResult:
        import mlx.core as mx
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_logits_processors, make_sampler

        msgs = messages_with_system(messages, system_prompt)
        prompt = self._encode(msgs)
        if self.context_window:
            limit = self.context_window - MIN_NEW_TOKENS
            while len(prompt) > limit:
                shorter = drop_oldest_turn(msgs)
                if shorter is None:
                    prompt = prompt[-limit:]
                    break
                msgs = shorter
                prompt = self._encode(msgs)
        tuned = tune_generation(params.max_new_tokens, len(prompt), context_window=self.context_window,
                                kv_bytes_per_token=None, free_bytes=None)
        sampling = params.temperature > 0
        sampler = make_sampler(temp=params.temperature, top_p=params.top_p if sampling else 0.0,
                               top_k=params.top_k if sampling else 0)
        processors = make_logits_processors(
            repetition_penalty=params.repetition_penalty if params.repetition_penalty != 1.0 else None)
        if params.seed is not None:
            mx.random.seed(params.seed)
        mx.reset_peak_memory()
        t0 = time.perf_counter()
        first: float | None = None
        completion = 0
        parts: list[str] = []
        finish = "stop"
        emitted = 0
        for response in stream_generate(self.model, self.tokenizer, prompt, max_tokens=tuned.max_new_tokens,
                                        sampler=sampler, logits_processors=processors):
            if first is None:
                first = time.perf_counter()
            completion = response.generation_tokens
            if response.text:
                parts.append(response.text)
                text = "".join(parts)
                if params.stop:
                    clipped, hit = _apply_stop_strings(text, params.stop)
                    if hit:
                        if len(clipped) > emitted:
                            on_text(clipped[emitted:])
                        parts = [clipped]
                        finish = "stop"
                        break
                on_text(text[emitted:])
                emitted = len(text)
            if stop_event.is_set():
                finish = "cancelled"
                break
            if response.finish_reason:
                finish = "length" if response.finish_reason == "length" else "stop"
        end = time.perf_counter()
        ttft, rate = _rates(t0, first, end, completion)
        return GenerationResult(
            text="".join(parts), prompt_tokens=len(prompt), completion_tokens=completion, ttft_ms=ttft,
            tokens_per_second=rate, total_ms=(end - t0) * 1000.0, peak_memory_bytes=int(mx.get_peak_memory()),
            finish_reason=finish,
            tuning={"max_new_tokens": tuned.max_new_tokens, "clamped": tuned.clamped, "reason": tuned.reason},
        )

    def complete(self, prompt: str, *, max_new_tokens: int = 96, system_prompt: str | None = None) -> str:
        params = GenerationParams(temperature=0.0, max_new_tokens=max_new_tokens, repetition_penalty=1.05)
        return self.stream_chat([{"role": "user", "content": prompt}], params, stop_event=threading.Event(),
                                on_text=lambda _t: None, system_prompt=system_prompt).text

    def release(self) -> None:
        self.model = None
        self.tokenizer = None
        try:
            import mlx.core as mx

            mx.clear_cache()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# llama.cpp (GGUF)
# ---------------------------------------------------------------------------


class LlamaCppTextEngine:
    engine = "llama_cpp"

    def __init__(self, llm: Any, n_ctx: int, device: str) -> None:
        self.llm = llm
        self.context_window = n_ctx
        self.device = device
        self.kv_per_token = None
        self.config: dict[str, Any] = {"n_ctx": n_ctx}

    @classmethod
    def load(cls, gguf_path: str, *, device: str, n_ctx: int = 4096) -> LlamaCppTextEngine:
        require("llama_cpp", "gguf")
        from llama_cpp import Llama

        if not Path(gguf_path).is_file():
            raise LoadFailed(f"GGUF file not found: {gguf_path}")
        llm = Llama(model_path=gguf_path, n_gpu_layers=-1 if device != "cpu" else 0, n_ctx=n_ctx, verbose=False)
        return cls(llm, n_ctx, device)

    def count_tokens(self, text: str) -> int:
        return len(self.llm.tokenize(text.encode("utf-8"), add_bos=False))

    def stream_chat(self, messages: list[dict[str, Any]], params: GenerationParams, *, stop_event: threading.Event,
                    on_text: OnText, system_prompt: str | None = None) -> GenerationResult:
        msgs = messages_with_system(messages, system_prompt)
        prompt_tokens = self.count_tokens(render_plain_prompt(msgs))
        tuned = tune_generation(params.max_new_tokens, prompt_tokens, context_window=self.context_window,
                                kv_bytes_per_token=None, free_bytes=None)
        t0 = time.perf_counter()
        first: float | None = None
        parts: list[str] = []
        completion = 0
        finish = "stop"
        stream = self.llm.create_chat_completion(
            messages=msgs, stream=True, temperature=params.temperature, top_p=params.top_p, top_k=params.top_k,
            repeat_penalty=params.repetition_penalty, max_tokens=tuned.max_new_tokens, stop=params.stop or None,
            seed=params.seed if params.seed is not None else -1)
        for chunk in stream:
            choice = chunk["choices"][0]
            delta = choice.get("delta", {}).get("content")
            if delta:
                if first is None:
                    first = time.perf_counter()
                completion += 1
                parts.append(delta)
                on_text(delta)
            if choice.get("finish_reason"):
                finish = "length" if choice["finish_reason"] == "length" else "stop"
            if stop_event.is_set():
                finish = "cancelled"
                break
        end = time.perf_counter()
        ttft, rate = _rates(t0, first, end, completion)
        return GenerationResult(text="".join(parts), prompt_tokens=prompt_tokens, completion_tokens=completion,
                                ttft_ms=ttft, tokens_per_second=rate, total_ms=(end - t0) * 1000.0,
                                peak_memory_bytes=None, finish_reason=finish,
                                tuning={"max_new_tokens": tuned.max_new_tokens, "clamped": tuned.clamped,
                                        "reason": tuned.reason})

    def complete(self, prompt: str, *, max_new_tokens: int = 96, system_prompt: str | None = None) -> str:
        params = GenerationParams(temperature=0.0, max_new_tokens=max_new_tokens)
        return self.stream_chat([{"role": "user", "content": prompt}], params, stop_event=threading.Event(),
                                on_text=lambda _t: None, system_prompt=system_prompt).text

    def release(self) -> None:
        if self.llm is not None:
            try:
                self.llm.close()
            except Exception:
                pass
        self.llm = None


# ---------------------------------------------------------------------------
# llama-server (GGUF via the downloaded llama.cpp component; works in frozen builds)
# ---------------------------------------------------------------------------


def _free_port() -> int:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _tail_text(path: Path, limit: int) -> str:
    """Last ``limit`` bytes of a log file without loading the whole file."""
    try:
        with path.open("rb") as fh:
            fh.seek(0, os.SEEK_END)
            fh.seek(max(fh.tell() - limit, 0))
            return fh.read(limit).decode("utf-8", "replace")
    except OSError:
        return ""


def _llama_server_help(binary: str) -> str:
    import subprocess

    kwargs: dict[str, Any] = {}
    if sys.platform == "win32":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
    try:
        result = subprocess.run([binary, "--help"], capture_output=True, text=True, timeout=30, **kwargs)
    except (OSError, subprocess.SubprocessError) as exc:
        raise LoadFailed(f"Could not start llama-server: {exc}") from exc
    return (result.stdout or "") + (result.stderr or "")


class LlamaServerEngine:
    """Runs a GGUF file in the official ``llama-server`` child process (loopback only) and streams
    from its OpenAI-compatible endpoint. Metal/CUDA offload is handled by llama.cpp itself."""

    engine = "llama_server"

    def __init__(self, process: Any, port: int, n_ctx: int, device: str, api_key: str = "") -> None:
        self.process = process
        self.port = port
        self.api_key = api_key
        self.context_window = n_ctx
        self.device = device
        self.kv_per_token = None
        self.config: dict[str, Any] = {"n_ctx": n_ctx}

    @classmethod
    def load(cls, gguf_path: str, *, device: str, binary: str, n_ctx: int = 4096,
             startup_timeout: float = 180.0) -> LlamaServerEngine:
        import subprocess

        from noelle import paths

        if not Path(gguf_path).is_file():
            raise LoadFailed(f"GGUF file not found: {gguf_path}")
        import secrets

        help_text = _llama_server_help(binary)
        if "--api-key" not in help_text:
            raise LoadFailed("This llama-server build is too old (no --api-key support).",
                             hint="Update the llama.cpp tools in Settings › Components.")
        port = _free_port()
        api_key = secrets.token_urlsafe(32)  # loopback is shared by every local process: require a key
        args = [binary, "-m", gguf_path, "--host", "127.0.0.1", "--port", str(port), "-c", str(n_ctx),
                "-ngl", "999" if device != "cpu" else "0"]
        if "--no-webui" in help_text:
            args.append("--no-webui")
        env = {k: v for k, v in os.environ.items() if k not in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")}
        if "LLAMA_API_KEY" in help_text:
            env["LLAMA_API_KEY"] = api_key  # keeps the key out of process listings
        else:
            args += ["--api-key", api_key]
        log_path = paths.logs_dir() / "llama-server.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log = open(log_path, "ab")  # noqa: SIM115 - owned by the child process
        kwargs: dict[str, Any] = {}
        if sys.platform == "win32":
            kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
        process = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT, env=env, **kwargs)
        log.close()
        engine = cls(process, port, n_ctx, device, api_key)
        deadline = time.monotonic() + startup_timeout
        while time.monotonic() < deadline:
            if process.poll() is not None:
                tail = _tail_text(log_path, 600).strip().splitlines()
                raise LoadFailed(f"llama-server exited (code {process.returncode}): {tail[-1] if tail else 'no output'}")
            try:
                status, _ = engine._call("GET", "/health", timeout=2)
                if status == 200:
                    return engine
            except OSError:
                pass
            time.sleep(0.25)
        engine.release()
        raise LoadFailed("llama-server did not become ready in time.")

    def _call(self, method: str, path: str, body: dict[str, Any] | None = None,
              timeout: float = 30) -> tuple[int, Any]:
        import http.client
        import json

        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=timeout)
        try:
            conn.request(method, path, body=json.dumps(body) if body is not None else None,
                         headers=self._headers())
            response = conn.getresponse()
            raw = response.read()
            try:
                return response.status, json.loads(raw) if raw else None
            except ValueError:
                return response.status, raw.decode("utf-8", "replace")
        finally:
            conn.close()

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def count_tokens(self, text: str) -> int:
        status, data = self._call("POST", "/tokenize", {"content": text})
        if status == 200 and isinstance(data, dict):
            return len(data.get("tokens") or [])
        return max(1, len(text) // 4)

    def stream_chat(self, messages: list[dict[str, Any]], params: GenerationParams, *, stop_event: threading.Event,
                    on_text: OnText, system_prompt: str | None = None) -> GenerationResult:
        import http.client
        import json

        msgs = messages_with_system(messages, system_prompt)
        prompt_tokens = self.count_tokens(render_plain_prompt(msgs))
        tuned = tune_generation(params.max_new_tokens, prompt_tokens, context_window=self.context_window,
                                kv_bytes_per_token=None, free_bytes=None)
        body: dict[str, Any] = {"messages": msgs, "stream": True, "max_tokens": tuned.max_new_tokens,
                                "temperature": params.temperature, "top_p": params.top_p, "top_k": params.top_k,
                                "repeat_penalty": params.repetition_penalty,
                                "stream_options": {"include_usage": True}}
        if params.seed is not None:
            body["seed"] = params.seed
        if params.stop:
            body["stop"] = params.stop
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=600)
        t0 = time.perf_counter()
        first: float | None = None
        parts: list[str] = []
        completion = 0
        finish = "stop"
        usage: dict[str, Any] = {}
        try:
            conn.request("POST", "/v1/chat/completions", body=json.dumps(body),
                         headers={**self._headers(), "Accept": "text/event-stream"})
            response = conn.getresponse()
            if response.status != 200:
                raise LoadFailed(f"llama-server returned HTTP {response.status}: "
                                 f"{response.read()[:300].decode('utf-8', 'replace')}")
            for raw in response:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                usage = chunk.get("usage") or usage
                for choice in chunk.get("choices") or []:
                    delta = (choice.get("delta") or {}).get("content")
                    if delta:
                        if first is None:
                            first = time.perf_counter()
                        completion += 1
                        parts.append(delta)
                        on_text(delta)
                    if choice.get("finish_reason"):
                        finish = "length" if choice["finish_reason"] == "length" else "stop"
                if stop_event.is_set():
                    finish = "cancelled"  # closing the connection makes llama-server stop generating
                    break
        finally:
            conn.close()
        end = time.perf_counter()
        completion = int(usage.get("completion_tokens") or completion)
        ttft, rate = _rates(t0, first, end, completion)
        return GenerationResult(text="".join(parts), prompt_tokens=int(usage.get("prompt_tokens") or prompt_tokens),
                                completion_tokens=completion, ttft_ms=ttft, tokens_per_second=rate,
                                total_ms=(end - t0) * 1000.0, peak_memory_bytes=None, finish_reason=finish,
                                tuning={"max_new_tokens": tuned.max_new_tokens, "clamped": tuned.clamped,
                                        "reason": tuned.reason})

    def complete(self, prompt: str, *, max_new_tokens: int = 96, system_prompt: str | None = None) -> str:
        params = GenerationParams(temperature=0.0, max_new_tokens=max_new_tokens)
        return self.stream_chat([{"role": "user", "content": prompt}], params, stop_event=threading.Event(),
                                on_text=lambda _t: None, system_prompt=system_prompt).text

    def release(self) -> None:
        process, self.process = self.process, None
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except Exception:  # noqa: BLE001 - escalate, then always reap so no zombie is left behind
                process.kill()
                try:
                    process.wait(timeout=5)
                except Exception:  # noqa: BLE001 - the OS owns it now; nothing else can be done here
                    pass


# ---------------------------------------------------------------------------
# Vision-language
# ---------------------------------------------------------------------------


class VisionLanguageEngine:
    """Image + prompt -> streamed text (VLMs, captioners and OCR models such as TrOCR)."""

    engine = "transformers-vlm"

    def __init__(self, model: Any, processor: Any, device: str, config: dict[str, Any], encoder_decoder: bool) -> None:
        self.model = model
        self.processor = processor
        self.device = device
        self.config = config
        self.encoder_decoder = encoder_decoder
        self.context_window = context_limit(config)

    @classmethod
    def load(cls, source: str, *, device: str, precision: str = "auto", trust_remote_code: bool = False,
             revision: str | None = None) -> VisionLanguageEngine:
        from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

        common: dict[str, Any] = {"trust_remote_code": trust_remote_code}
        if revision:
            common["revision"] = revision
        config = AutoConfig.from_pretrained(source, **common)
        processor = AutoProcessor.from_pretrained(source, **common)
        model_kwargs = _model_kwargs_for(device, precision)
        if config.model_type == "vision-encoder-decoder":
            from transformers import VisionEncoderDecoderModel

            model = VisionEncoderDecoderModel.from_pretrained(source, **common, **model_kwargs)
            encoder_decoder = True
        else:
            model = AutoModelForImageTextToText.from_pretrained(source, **common, **model_kwargs)
            encoder_decoder = False
        model.eval()
        return cls(model, processor, device, config.to_dict(), encoder_decoder)

    def _inputs(self, image: Any, prompt: str) -> dict[str, Any]:
        if self.encoder_decoder:
            inputs = dict(self.processor(images=image, return_tensors="pt"))
        else:
            content = [{"type": "image"}, {"type": "text", "text": prompt or "Describe this image in detail."}]
            text = prompt
            if getattr(self.processor, "chat_template", None):
                text = self.processor.apply_chat_template([{"role": "user", "content": content}],
                                                          add_generation_prompt=True, tokenize=False)
            inputs = dict(self.processor(text=[text], images=[image], return_tensors="pt"))
        dtype = getattr(self.model, "dtype", None)
        moved: dict[str, Any] = {}
        for key, value in inputs.items():
            if hasattr(value, "to"):
                value = value.to(self.model.device)
                if dtype is not None and value.is_floating_point():
                    value = value.to(dtype)
            moved[key] = value
        return moved

    def stream(self, image_path: str, prompt: str, params: GenerationParams, *, stop_event: threading.Event,
               on_text: OnText) -> GenerationResult:
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        inputs = self._inputs(image, prompt)
        prompt_tokens = int(inputs["input_ids"].shape[-1]) if "input_ids" in inputs else 0
        tuned = tune_generation(params.max_new_tokens, prompt_tokens, context_window=self.context_window,
                                kv_bytes_per_token=None, free_bytes=None)
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        helper = TransformersTextEngine.__new__(TransformersTextEngine)
        helper.model = self.model
        helper.tokenizer = tokenizer
        helper.device = self.device
        return helper._generate(inputs, prompt_tokens, params, tuned, stop_event, on_text, tokenizer=tokenizer)

    def release(self) -> None:
        self.model = None
        self.processor = None
