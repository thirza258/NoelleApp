"""PEFT / LoRA adapter hot-swapping on a resident base model.

Adapters are attached with transformers' built-in PEFT integration
(``model.load_adapter``); switching calls ``model.set_adapter`` which only flips
which low-rank matrices are active, so the base weights stay in memory.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from noelle.runtime.errors import BadRequest, ModelNotFound, Unsupported, require
from noelle.runtime.generation import TransformersTextEngine
from noelle.runtime.model_manager import LoadedModel


def _resolve_adapter(adapter: str, local_path: str | None) -> str:
    if local_path:
        path = Path(local_path).expanduser()
        if not (path / "adapter_config.json").is_file():
            raise ModelNotFound(f"No adapter_config.json in {path}")
        return str(path)
    from noelle.runtime.hub_cache import cached_snapshot_path

    snapshot = cached_snapshot_path(adapter)
    if snapshot is None or not (snapshot / "adapter_config.json").is_file():
        raise ModelNotFound(f"Adapter '{adapter}' is not downloaded.", hint="Download it from its Hub page first.")
    return str(snapshot)


def adapter_base_model(adapter_dir: str) -> str | None:
    try:
        data = json.loads((Path(adapter_dir) / "adapter_config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data.get("base_model_name_or_path")


def _engine(loaded: LoadedModel) -> TransformersTextEngine:
    if not isinstance(loaded.engine, TransformersTextEngine):
        raise Unsupported(f"LoRA hot-swapping needs the transformers engine; {loaded.spec.model_id} runs on "
                          f"{loaded.engine_name}.")
    return loaded.engine


def load_adapter(loaded: LoadedModel, adapter_id: str, *, name: str | None = None,
                 local_path: str | None = None, activate: bool = True) -> dict[str, Any]:
    require("peft", "peft")
    engine = _engine(loaded)
    adapter_dir = _resolve_adapter(adapter_id, local_path)
    name = name or adapter_id.split("/")[-1]
    if name in loaded.adapters:
        raise BadRequest(f"An adapter named '{name}' is already attached.")
    base = adapter_base_model(adapter_dir)
    t0 = time.perf_counter()
    engine.model.load_adapter(adapter_dir, adapter_name=name)
    load_ms = (time.perf_counter() - t0) * 1000.0
    loaded.adapters.append(name)
    switch_ms = None
    if activate:
        switch_ms = set_adapter(loaded, name)["switch_ms"]
    else:
        # load_adapter activates the new adapter; restore the previous state
        set_adapter(loaded, loaded.active_adapter)
    return {"adapter": name, "base_model": base, "load_ms": load_ms, "switch_ms": switch_ms,
            "adapters": list(loaded.adapters), "active_adapter": loaded.active_adapter,
            "base_matches": (base is None or base == loaded.spec.model_id)}


def set_adapter(loaded: LoadedModel, name: str | None) -> dict[str, Any]:
    engine = _engine(loaded)
    model = engine.model
    t0 = time.perf_counter()
    if name is None:
        if loaded.adapters:
            model.disable_adapters()
    else:
        if name not in loaded.adapters:
            raise BadRequest(f"Unknown adapter '{name}'. Attached: {', '.join(loaded.adapters) or 'none'}")
        if loaded.active_adapter is None and loaded.adapters:
            model.enable_adapters()
        model.set_adapter(name)
    switch_ms = (time.perf_counter() - t0) * 1000.0
    loaded.active_adapter = name
    return {"active_adapter": name, "switch_ms": switch_ms, "adapters": list(loaded.adapters)}


def list_adapters(loaded: LoadedModel) -> dict[str, Any]:
    return {"adapters": list(loaded.adapters), "active_adapter": loaded.active_adapter}
