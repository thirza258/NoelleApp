"""Runtime adapter contracts and the polymorphic factory."""

from .base_adapter import BaseModelAdapter, UnsupportedArchitectureError
from .factory import AdapterSpec, get_adapter_for_model, get_adapter_spec

__all__ = ["AdapterSpec", "BaseModelAdapter", "UnsupportedArchitectureError", "get_adapter_for_model",
           "get_adapter_spec"]