"""Optional native engines: Kokoro TTS, faster-whisper ASR, Model2Vec and single-file diffusion."""

from .faster_whisper_engine import FasterWhisperEngine
from .kokoro_engine import KokoroEngine
from .model2vec_engine import Model2VecEngine
from .registry import NativeEngineStatus, detect_native_engines
from .single_file_diffusion import SingleFileDiffusionEngine

__all__ = ["FasterWhisperEngine", "KokoroEngine", "Model2VecEngine", "NativeEngineStatus", "SingleFileDiffusionEngine",
           "detect_native_engines"]