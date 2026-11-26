"""Abstract execution contract for modality adapters.

The existing engines in :mod:`noelle.runtime.generation` / :mod:`pipelines` already
follow this shape (``load`` / ``release``); new native engines register an
:class:`~noelle.runtime.adapters.factory.AdapterSpec` and are exposed through the
same worker commands. This base class is for engines implemented as classes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generator


class UnsupportedArchitectureError(RuntimeError):
    """The factory cannot pair this repository with any engine Noelle has."""


class BaseModelAdapter(ABC):
    family: str = ""
    engine: str = ""

    @abstractmethod
    def load(self, model_path: str, device: str = "auto", precision: str = "auto", **kwargs: Any) -> None:
        ...

    @abstractmethod
    def run_inference(self, payload: dict[str, Any]) -> Any:
        ...

    @abstractmethod
    def stream_inference(self, payload: dict[str, Any]) -> Generator[Any, None, None]:
        ...

    @abstractmethod
    def unload(self) -> None:
        ...

    @abstractmethod
    def get_memory_footprint(self) -> dict[str, Any]:
        ...