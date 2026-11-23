"""Official ``openai`` client -> Noelle API server (uvicorn thread) -> real worker -> SmolLM2-135M on this machine."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pytest

from noelle.app.services.model_registry import ModelRecord, ModelRegistry
from noelle.runtime import server_service as srv

openai = pytest.importorskip("openai")

pytestmark = pytest.mark.integration

TEXT_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
MESSAGES = [{"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Name three colours."}]


@pytest.fixture
def api(worker_process, text_model_path: Path, tmp_path: Path) -> Iterator[openai.OpenAI]:
    registry = ModelRegistry(tmp_path / "registry.db")
    registry.upsert_model(ModelRecord(repo_id=TEXT_MODEL, pipeline_tag="text-generation",
                                      local_path=str(text_model_path)))
    server = srv.serve_in_thread(srv.create_app(worker_process.endpoint, worker_process.token, registry=registry))
    client = openai.OpenAI(base_url=f"{server.base_url}/v1", api_key="not-needed", max_retries=0, timeout=300)
    try:
        yield client
    finally:
        client.close()
        server.stop()
        registry.close()


def test_models_endpoint(api: openai.OpenAI) -> None:
    models = api.models.list()
    ids = [m.id for m in models.data]
    assert ids == [TEXT_MODEL]
    assert models.data[0].owned_by == "local"
    assert models.data[0].model_extra.get("task") == "text-generation"


def test_streaming_chat_completion(api: openai.OpenAI) -> None:
    stream = api.chat.completions.create(model=TEXT_MODEL, messages=MESSAGES, stream=True, max_tokens=24,
                                         temperature=0, stream_options={"include_usage": True})
    roles, pieces, finish, usage = [], [], [], None
    for chunk in stream:
        if chunk.usage is not None:
            usage = chunk.usage
            assert chunk.choices == []
            continue
        choice = chunk.choices[0]
        if choice.delta.role:
            roles.append(choice.delta.role)
        if choice.delta.content:
            pieces.append(choice.delta.content)
        if choice.finish_reason:
            finish.append(choice.finish_reason)
    assert roles == ["assistant"]
    assert len(pieces) >= 2 and "".join(pieces).strip()
    assert len(finish) == 1 and finish[0] in ("stop", "length")
    assert usage is not None
    assert usage.prompt_tokens > 5
    assert 0 < usage.completion_tokens <= 24
    assert usage.total_tokens == usage.prompt_tokens + usage.completion_tokens
    if finish[0] == "length":
        assert usage.completion_tokens == 24


def test_non_streaming_chat_completion(api: openai.OpenAI) -> None:
    response = api.chat.completions.create(model=TEXT_MODEL, messages=MESSAGES, max_tokens=8, temperature=0,
                                           seed=1)
    assert response.object == "chat.completion"
    assert response.model == TEXT_MODEL
    choice = response.choices[0]
    assert choice.message.role == "assistant"
    assert choice.message.content and choice.message.content.strip()
    assert choice.finish_reason in ("stop", "length")
    assert 0 < response.usage.completion_tokens <= 8
    assert response.usage.total_tokens == response.usage.prompt_tokens + response.usage.completion_tokens


def test_unknown_model_raises_not_found(api: openai.OpenAI) -> None:
    with pytest.raises(openai.NotFoundError) as info:
        api.chat.completions.create(model="nobody/nothing", messages=MESSAGES, max_tokens=4)
    assert TEXT_MODEL in str(info.value)
