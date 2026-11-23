---
title: Troubleshooting
nav: Troubleshooting
description: Fixes for common problems — out of memory, gated models, blocked remote code, GGUF tools, slow generation, installer warnings and API errors.
section: Reference
order: 31
slug: troubleshooting
---

## “Out of memory” when loading or generating

The model needs more memory than is free. Pick the **Low memory** profile or a 4-bit build in the **Analysis** tab,
close other models (only one large model fits on most laptops) and quit memory-hungry apps. The runtime recovers
on its own; if it was stopped, click **Restart runtime** in the banner.

## “The model runtime stopped unexpectedly”

Click **Restart runtime**. **Show log** displays the last lines the runtime wrote. Logs are in `~/.noelle/logs/`
(`app.log`, `worker-stderr.log`, `server-access.log`, `mcp.log`).

## A model says it is gated (401 or 403)

Open the model on huggingface.co, accept its licence with your account, then add a Hugging Face token in
**Settings › Account** (a read token is enough).

## “Remote code blocked”

The repository ships Python code that runs when the model loads. Review it, then choose **Allow once**,
**Trust repository** or **Trust publisher** in the Workspace sidebar. See
[Models & builds](docs:models#repositories-that-run-their-own-code).

## GGUF models need llama.cpp

Noelle offers to download the official llama.cpp tools when you first need them. You can also install, update or
remove them in **Settings › Components**.

## Generation is slow

Check the model's badge and the **Analysis** tab: a red badge or “CPU fallback” means the model doesn't fit on your
GPU. On a Mac, a **4-bit MLX** build is usually several times faster than full precision; on a PC without a
strong GPU, a **GGUF** build is the best choice.

## macOS: “Noelle can't be opened”

Open **System Settings › Privacy & Security**, scroll down and click **Open Anyway**. You only need to do this once.

## Windows: “Windows protected your PC”

Click **More info › Run anyway**.

## API errors

| Status | Cause | Fix |
|---|---|---|
| 401 | An API key is set | Send `Authorization: Bearer <key>` |
| 404 `model_not_found` | Unknown model or alias | Use a name from `GET /v1/models` |
| 415 | Body not sent as JSON | Add `Content-Type: application/json` |
| 421 | Request addressed to another host name | Use `http://127.0.0.1:8765` or `http://localhost:8765` |
| 503 `worker_unavailable` | The model runtime isn't running | Open Noelle and start or restart the runtime |

## MCP: “Noelle's API server is not running”

Start the server on Noelle's **API** page; the MCP tools `chat`, `embed` and the knowledge tools use it.
