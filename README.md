# Noelle Desktop AI Studio

Noelle is a local-first desktop workbench for open-weight models from the Hugging Face Hub. You can discover a model,
check whether it fits your hardware, download it with checksum verification, run it in a playground for its modality,
compare several models on one prompt, index your own documents for grounded answers, and serve everything through a
local OpenAI-compatible API. Inference, document indexing and audio processing all run on your machine.

Version 1.0.0 replaces the previous Django + React web app that lived in this repository.

## Highlights

| Area | What you get |
|---|---|
| **Discover** | Live Hub catalog by modality (text, vision, audio, diffusion, embeddings) with search, sorting and infinite scroll. A trending hero collapses (250 ms) when you scroll, so the grid gets the full width. Every card has a hardware badge: **green** runs natively, **yellow** needs a 4-bit build, **red** CPU fallback or too large. |
| **Workspace** | Model facts, a per-precision memory table, resumable verified downloads, and runtime controls for precision, device and quantized builds. Playgrounds for chat (streaming with TTFT and tok/s), vision Q&A, detection, classification, speech recognition (file or microphone), speech synthesis, text-to-image with live latent previews, and embeddings. |
| **Grounding** | Local RAG covers PDF, DOCX, Markdown and text, with 512/64 token-aware chunks, on-device embeddings and sqlite-vec search, and shows citations. Optional web grounding goes through your own SearXNG instance. |
| **Optimize** | One-click 4-bit quantization: MLX on Apple Silicon; AWQ, GPTQ and bitsandbytes NF4 on CUDA; GGUF with llama.cpp tools. LoRA adapters attach to a loaded model and switch without reloading the base weights. |
| **Compare** | 1, 2 or 4 views with a broadcast prompt. A sequential GPU queue runs View 1 → View 4 and either keeps every model loaded (retain) or unloads each before the next (hot-swap, with `gc.collect()` and the cache flushed). Produces a benchmark matrix (TTFT, tok/s, peak memory, precision) with JSON and Markdown export. |
| **Developer** | OpenAI-compatible server on `127.0.0.1` for `/v1/chat/completions` (SSE), `/v1/embeddings`, `/v1/audio/transcriptions` and `/v1/models`. Also exports a standalone `inference.py`, a CUDA `Dockerfile` and `server.py`, and includes an embedded Jupyter console (`noelle.load_model(...)`). |
| **Storage** | Hugging Face cache visualizer showing repos, revisions and partial files. You can prune unfinished downloads or detached revisions, delete repos, and move the cache to another drive. |

## Architecture

```
PySide6 desktop shell  (never imports torch)
 ├─ Discover · Workspace · Compare · Storage · API Server pages
 ├─ services: HF client (keyring token) · hardware telemetry · download manager · registry (SQLite WAL)
 └─ IPC: ZeroMQ DEALER on a dedicated QThread
        │  tcp://127.0.0.1:<random port>, per-session auth token
        ▼
Inference worker  (`noelle worker`, separate process)
 ├─ I/O thread: ROUTER socket, heartbeats, light commands
 └─ single GPU lane: every model op runs here in order → no oversubscription
      engines: transformers (CUDA/MPS/CPU) · MLX · llama.cpp · diffusers · sentence-transformers
      model manager: memory pre-flight, LRU eviction, OOM containment

`noelle server`    FastAPI/uvicorn, OpenAI API → same worker
`noelle download`  one child per repo: JSON-line progress, retries, SHA-256 verification
```

* **Fault isolation.** If a memory error happens inside the worker, the allocator cache is flushed and the request
  returns `ERROR{code: OOM}`. If the OS kills the worker, the shell sees the process exit, fails the pending requests
  and offers a restart. The UI keeps running either way.
* **Data locality.** The worker runs with `HF_HUB_OFFLINE=1` and receives no Hub token. The UI talks to the Hub only
  for metadata and downloads. Everything binds to loopback.

## Install

| OS | Installer | How |
|---|---|---|
| macOS 13+ | `Noelle-1.0.0-macos-arm64.pkg` (or the `.dmg`) | Double-click and follow the steps. "Install for me only" needs no administrator password. |
| Windows 10/11 | `Noelle-1.0.0-windows-x64-setup.exe` | Step-by-step wizard. It includes an optional page for your Hugging Face token. |
| Linux (Debian/Ubuntu) | `noelle_1.0.0_amd64.deb` | `sudo apt install ./noelle_1.0.0_amd64.deb`, then open *Noelle AI Studio* from the app menu. |

Download the installers from the repository's Releases page, or from the artifacts of the `release` workflow
(`gh workflow run release.yml`, then `gh run download`). On first launch the **Setup Assistant** walks you through:

1. Hugging Face token (optional). It is stored in the OS credential vault.
2. A hardware check: what your GPU and memory can run.
3. Optional downloads: llama.cpp tools for GGUF and a starter chat model. **Nothing downloads unless you tick it.**

Run the assistant again any time from *Settings › About*, or with `noelle setup`.

## Run from source

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[all,dev]"     # PySide6, torch, transformers, diffusers, RAG, server, MLX (Apple)
.venv/bin/python -m noelle                # launch the studio
```

Requires Python ≥ 3.11 (developed on 3.14, PySide6 6.12, torch 2.15, transformers 5).
On Linux/Windows with NVIDIA GPUs, install the CUDA build of PyTorch first.

### Command line

| Command | Purpose |
|---|---|
| `noelle` | Desktop studio |
| `noelle worker --port 0` | Inference worker (normally spawned by the studio) |
| `noelle server --port 8765 --worker-endpoint tcp://127.0.0.1:<p>` | OpenAI-compatible server |
| `noelle download <repo> [--include PATTERN] [--local-dir DIR]` | Verified download with JSON progress |
| `noelle env create <name> --profile base --profile rag` | Isolated runtime environment |

### Using the API from other tools

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8765/v1", api_key="not-needed")
for chunk in client.chat.completions.create(model="HuggingFaceTB/SmolLM2-135M-Instruct",
                                            messages=[{"role": "user", "content": "Hi"}], stream=True):
    print(chunk.choices[0].delta.content or "", end="")
```

The API Server page has copy-ready snippets for curl, Cursor, Continue and LangChain.

## Data locations

| Path | Contents |
|---|---|
| `~/.noelle/` (override: `NOELLE_HOME`) | `registry.db`, `settings.json`, `models/` (quantized or derived builds), `environments/`, `vector_store/`, `projects/` (outputs and exports), `logs/` |
| `~/.cache/huggingface/hub` (or `HF_HOME`) | Hub downloads. Stays deduplicated and is shared with other tools. Switch to explicit `~/.noelle/models` folders in Settings › Storage. |
| OS credential store | Hugging Face token (Keychain / Credential Manager / Secret Service), never in files |

## Hardware support

| Feature | NVIDIA CUDA | Apple Silicon (Metal) | CPU |
|---|---|---|---|
| FP16/BF16 inference | ✓ | ✓ (MPS) | ✓ (FP32) |
| 4-bit MLX quantize + run | — | ✓ | — |
| AWQ / GPTQ / NF4 (bitsandbytes) | ✓ | disabled, with the reason shown | — |
| GGUF quantize + run | ✓ | ✓ (Metal) | ✓ |

GGUF uses the official llama.cpp tools (`llama-quantize`, `llama-server` and the converter). Noelle offers to
download them the first time you need them, in Optimize, when loading a GGUF model, in *Settings › Components* or in
the Setup Assistant. It shows the version, size, source and destination and asks first, then verifies the SHA-256
published by GitHub. `llama-cpp-python` is used instead when it is installed.

## Tests

```bash
.venv/bin/python -m pytest tests/unit                 # fast, no models needed
QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest tests/e2e
.venv/bin/python -m pytest tests/integration          # uses small models already in your HF cache, skips otherwise
```

Tests isolate `NOELLE_HOME`, replace the OS keychain with an in-memory backend, and never modify the real Hugging Face
cache. Storage tests build fake caches in temporary directories.

## Packaging

`python packaging/build.py [--sign]` builds the PyInstaller app and the installer for the current OS:
`.pkg` + `.dmg` (macOS, via `pkgbuild`/`productbuild`/`hdiutil`), `-setup.exe` (Windows, Inno Setup) or `.deb` +
`.tar.gz` (Linux, `dpkg-deb`). The same binary also runs as the worker, server and download child. Signing reads `NOELLE_MAC_SIGN_IDENTITY` /
`NOELLE_NOTARY_PROFILE` (macOS) or `NOELLE_WIN_CERT_PATH` / `NOELLE_WIN_CERT_PASSWORD` (Windows) from the
environment. `.github/workflows/release.yml` builds macOS, Linux and Windows artifacts on `v*` tags.

## License

MIT, see [LICENSE](LICENSE).
