"""LoRA adapters attach to the resident base model and switch without reloading it (task T-505)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from noelle.runtime.protocol import Command, Event

pytestmark = pytest.mark.integration
BASE = "HuggingFaceTB/SmolLM2-135M-Instruct"


def _make_adapter(base_path: Path, out: Path, seed: int) -> Path:
    torch = pytest.importorskip("torch")
    peft = pytest.importorskip("peft")
    from transformers import AutoModelForCausalLM

    torch.manual_seed(seed)
    model = AutoModelForCausalLM.from_pretrained(str(base_path), dtype=torch.float32)
    config = peft.LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"], init_lora_weights=False,
                             task_type="CAUSAL_LM")
    peft.get_peft_model(model, config).save_pretrained(str(out))
    return out


def _generate(client: Any, adapter: str | None) -> str:
    payload = {"model_id": BASE, "messages": [{"role": "user", "content": "Name a planet."}],
               "params": {"temperature": 0, "max_new_tokens": 8}, "adapter": adapter}
    final = list(client.stream(Command.GENERATE_STREAM, payload))[-1]
    assert final["event"] == Event.GENERATION_COMPLETE, final
    assert final["payload"]["metrics"]["adapter"] == adapter
    return final["payload"]["text"]


def test_attach_and_hot_swap_adapters(worker_process: Any, text_model_path: Path, tmp_path: Path) -> None:
    first = _make_adapter(text_model_path, tmp_path / "lora-a", seed=1)
    second = _make_adapter(text_model_path, tmp_path / "lora-b", seed=2)
    client = worker_process.client()
    client.call(Command.LOAD_MODEL, {"model_id": BASE})
    loaded_before = client.call(Command.LIST_LOADED)["models"]
    load_ms_before = next(m["load_ms"] for m in loaded_before if m["model_id"] == BASE)

    a = client.call(Command.LORA_LOAD, {"model_id": BASE, "adapter_id": "lora-a", "adapter_local_path": str(first)})
    b = client.call(Command.LORA_LOAD, {"model_id": BASE, "adapter_id": "lora-b", "adapter_local_path": str(second)})
    assert a["adapters"] == ["lora-a"] and b["adapters"] == ["lora-a", "lora-b"] and b["active_adapter"] == "lora-b"

    switch_times = []
    for name in ("lora-a", "lora-b", None, "lora-a"):
        result = client.call(Command.LORA_SET_ADAPTER, {"model_id": BASE, "adapter_name": name})
        assert result["active_adapter"] == name
        switch_times.append(result["switch_ms"])
    assert max(switch_times) < 50, switch_times

    base_text = _generate(client, None)
    with_a = _generate(client, "lora-a")
    with_b = _generate(client, "lora-b")
    assert len({base_text, with_a, with_b}) >= 2  # random adapters change the output

    loaded_after = client.call(Command.LIST_LOADED)["models"]
    entry = next(m for m in loaded_after if m["model_id"] == BASE)
    assert entry["load_ms"] == load_ms_before  # base weights were never reloaded
    assert entry["adapters"] == ["lora-a", "lora-b"]
