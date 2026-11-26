"""Model2Vec static embeddings (optional native component, very fast on CPU)."""

from __future__ import annotations

from typing import Any

import numpy as np

from noelle.runtime.errors import DependencyMissing


class Model2VecEngine:
    engine = "model2vec"

    def __init__(self, model: Any, device: str = "cpu") -> None:
        self.model = model
        self.device = device
        self.model_type = "model2vec"
        self.tokenizer = None

    @classmethod
    def load(cls, source: str, *, device: str = "auto", **kwargs: Any) -> Model2VecEngine:
        try:
            from model2vec import StaticModel
        except ImportError as exc:
            raise DependencyMissing("Model2Vec embeddings need the 'model2vec' package.",
                                    hint='Install the "model2vec" component in Settings › Components.',
                                    component="model2vec") from exc
        return cls(StaticModel.from_pretrained(source), "cpu")

    def encode(self, texts: list[str], *, batch_size: int = 32, normalize: bool = True, **kwargs: Any) -> np.ndarray:
        vectors = self.model.encode(list(texts), show_progress_bar=False)
        vectors = np.asarray(vectors, dtype=np.float32)
        if normalize:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            vectors = vectors / np.maximum(norms, 1e-12)
        return vectors

    @property
    def dimension(self) -> int:
        return int(self.encode(["dimension probe"]).shape[1])

    @property
    def max_seq_length(self) -> int:
        return int(getattr(self.model, "max_seq_length", 512) or 512)

    def release(self) -> None:
        self.model = None