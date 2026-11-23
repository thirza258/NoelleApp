---
title: Local API
nav: Local API
description: Use your local models from any OpenAI-compatible client, with aliases, automatic routing, image generation and knowledge search.
section: Developers
order: 20
slug: api
---

Noelle includes an **OpenAI-compatible API server** so your scripts, apps and tools can use your local models.
Open the **API** page and click **Start server**. It listens on `http://127.0.0.1:8765/v1` by default.

## Quick example

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8765/v1", api_key="not-needed")
stream = client.chat.completions.create(
    model="HuggingFaceTB/SmolLM2-360M-Instruct",
    messages=[{"role": "user", "content": "Write a haiku about rain."}],
    stream=True,
)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="")
```

The API page has ready-to-copy snippets for curl, Python, Cursor, Continue and LangChain.

## Endpoints

| Method | Path | |
|---|---|---|
| GET | `/v1/models` | Downloaded models, your aliases and `auto` |
| POST | `/v1/chat/completions` | Chat, with streaming (Server-Sent Events) |
| POST | `/v1/embeddings` | Vectors from a local embedding model |
| POST | `/v1/audio/transcriptions` | Speech to text from an uploaded file |
| POST | `/v1/images/generations` | Text to image, returned as base64 (`b64_json`) |
| GET | `/v1/knowledge` | Your knowledge bases |
| POST | `/v1/knowledge/query` | Search a knowledge base: `{"collection", "query", "top_k"}` |
| GET | `/health` | Server and runtime status |

The model name is the one shown on the API page (usually the Hugging Face repository id). Local builds are listed
with their build name.

## Aliases

Aliases are stable names that point at a model — for example `noelle-chat` or `noelle-code`. Your apps use the alias,
and you can switch the model behind it on the API page without changing any code. Responses report the model that
actually answered, also in the `x-noelle-model` header.

## Automatic routing: `"model": "auto"`

With `"model": "auto"`, Noelle picks a downloaded model for each request:

| Request | Order of preference |
|---|---|
| Chat that looks like code | the `noelle-code` alias |
| Other chat | the `noelle-chat` alias, a model that is already loaded, the most recently used text model |
| Embeddings | the `noelle-embed` alias, a loaded embedding model, your knowledge-base embedding model, the most recent one |
| Images | the `noelle-image` alias, the most recent image model |

The `x-noelle-routed-model` and `x-noelle-route-reason` headers tell you what was chosen and why.

## Images

```bash
curl http://127.0.0.1:8765/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{"model": "auto", "prompt": "a lighthouse at dusk, watercolor", "size": "512x512", "seed": 7}'
```

`size` must be `WIDTHxHEIGHT` with both sides multiples of 8 between 256 and 2048. Only `"n": 1` and
`"response_format": "b64_json"` are supported; `steps` and `seed` are optional extras.

## Security

- The server only listens on this computer (`127.0.0.1`).
- It answers only requests addressed to `127.0.0.1`, `localhost` or `[::1]`, which stops websites from reaching it
  through DNS tricks, and JSON endpoints require `Content-Type: application/json`.
- Set an **API key** on the API page to require `Authorization: Bearer <key>` from every client, including other
  programs on this computer.
