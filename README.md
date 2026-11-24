# Noelle Desktop AI Studio

Noelle is a local-first desktop workbench — a small model lab — for open-weight models from the Hugging Face Hub. Tell
it what you want to build and it proposes a stack that fits your hardware; or discover a model yourself, see a 0–100
score for how well it will run here and the best way to run it, download it with checksum verification, try it in a
playground, evaluate several models on a dataset, index your own documents for grounded answers, and serve everything
through a local OpenAI-compatible API or to coding agents over MCP. Inference, document indexing and audio processing all
run on your machine.

Version 1.0.0 replaces the previous Django + React web app that lived in this repository.

## Highlights

| Area | What you get |
|---|---|
| **Mission** | Describe a goal ("a local Indonesian legal assistant for my PDF contracts", "…with 16 GB RAM and an RTX 3060"). Noelle detects the task, languages and stated hardware, searches the Hub, scores every candidate for that hardware and proposes a stack (LLM + embedding model + vector store) with memory estimates and alternatives. It then runs the plan step by step — download → build → index → test → serve — asking before anything is downloaded, converted or started. |
| **Discover** | Live Hub catalog by modality (text, vision, audio, diffusion, embeddings) with search, sorting and infinite scroll. A trending hero collapses (250 ms) when you scroll, so the grid gets the full width. Every card shows a 0–100 compatibility score with a badge: **green** runs natively, **yellow** needs a 4-bit build, **red** CPU fallback or too large. |
| **Workspace** | **Analysis** tab: the score broken into memory fit, runtime, precision, context and trust; "How should I run this?" — every build (FP16, INT8, NF4, AWQ, GPTQ, 4-bit MLX, GGUF) with memory, fit, quality, speed (measured on this machine when available, otherwise a labelled estimate) and whether it is ready, needs a one-time build, an optional component or cannot run here; the largest context that fits; capabilities (inputs/outputs, chat template, tool calling, languages, licence, remote code). **Runtime profiles** (Balanced, Quality, Fast, Low memory, Max context) pick the build for you. Repositories with their own Python code show the files and offer *Allow once*, *Trust repository* or *Trust publisher*. Plus provenance (revision, download date), resumable verified downloads and playgrounds for chat, vision Q&A, detection, classification, speech recognition, speech synthesis, text-to-image and embeddings. |
| **Evaluate** | The Evaluation Lab: run built-in or your own JSONL datasets (reasoning, JSON output, instruction following, sentiment, coding, Indonesian Q&A, EN→ID translation, chat latency) against several local models. Cases are scored locally and deterministically (exact, contains, regex, numeric, choice, JSON checks — generated code is never executed). Every run is saved as an experiment with the dataset digest, generation settings, hardware and each model's revision, with history, JSON/Markdown export and a performance matrix built from your own runs. |
| **Knowledge** | RAG Studio: knowledge bases built from PDF, DOCX, Markdown, text and code files, with token-aware chunks (512/64 by default, adjustable per base), on-device embeddings and sqlite-vec search with citations. A retrieval **evaluation** tab scores a set of questions (hit rate, recall, precision, MRR and nDCG at k) and compares two bases side by side — e.g. two chunk sizes or two embedding models. Chat in the Workspace grounds answers in a base, optionally with web results from your own SearXNG instance. |
| **Optimize** | One-click 4-bit quantization: MLX on Apple Silicon; AWQ, GPTQ and bitsandbytes NF4 on CUDA; GGUF with llama.cpp tools. LoRA adapters attach to a loaded model and switch without reloading the base weights. |
| **Compare** | 1, 2 or 4 views with a broadcast prompt. A sequential GPU queue runs View 1 → View 4 and either keeps every model loaded (retain) or unloads each before the next (hot-swap, with `gc.collect()` and the cache flushed). Produces a benchmark matrix (TTFT, tok/s, peak memory, precision) with JSON and Markdown export; runs also appear in the Evaluation Lab history. |
| **Developer** | Local AI gateway on `127.0.0.1`: OpenAI-compatible `/v1/chat/completions` (SSE), `/v1/embeddings`, `/v1/audio/transcriptions`, `/v1/images/generations` and `/v1/models`, plus `/v1/knowledge` search. Model **aliases** (`noelle-chat`, `noelle-code`, …) and `model: "auto"` routing (code → code model, embeddings → embedding model, …). `noelle mcp` exposes the same to coding agents. Also exports a standalone `inference.py`, a CUDA `Dockerfile` and `server.py`, and includes an embedded Jupyter console (`noelle.load_model(...)`). |
| **Storage** | Hugging Face cache visualizer showing repos, revisions and partial files, with a **Storage Advisor**: active vs unused (90+ days) models, partial downloads, old revisions, local builds and the space a cleanup would free. Prune, delete repos, and move the cache to another drive. |
| **Appearance** | Light (white) and Dark (neutral grey) themes, or follow the operating system (Settings › Appearance). |

## Architecture

```
PySide6 desktop shell  (never imports torch)
 ├─ Discover · Mission · Workspace · Knowledge · Evaluate · Compare · Storage · API pages
 ├─ model intelligence (pure Python, shared with CLI/MCP): capability registry · runtime/quantization advisor
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

Download the installers from the repository's Releases page, or from [SourceForge](https://sourceforge.net/projects/noelleapp/)
(or from the artifacts of the `release` workflow with `gh workflow run release.yml`, then `gh run download`). On first launch the **Setup Assistant** walks you through:

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
| `noelle env export <name> -o noelle-env.yaml` / `noelle env import noelle-env.yaml` | Pinned lockfile to rebuild an environment exactly |
| `noelle analyze <repo> [--profile fast] [--json]` | Compatibility score, run options and recommendation for this machine |
| `noelle run -m <model> "prompt"` | One prompt to a local model, streamed to stdout (`--metrics` for TTFT and tok/s) |
| `noelle eval -m <model> -d builtin:reasoning-math [-t accuracy>=0.8]` | Evaluation Lab from the terminal; exit code 1 when a threshold or `--baseline` check fails |
| `noelle mcp [--print-config]` | Model Context Protocol server on stdio |

### Evaluation Lab in CI

`noelle eval` runs datasets in a private worker, saves the run in the experiment history and prints a leaderboard.
Thresholds and a previous run turn it into a regression gate:

```bash
noelle eval --list-datasets
noelle eval -m HuggingFaceTB/SmolLM2-360M-Instruct -d builtin:reasoning-math -d ./my-cases.jsonl \
    -t "accuracy>=0.6" -t "tok_s>=20" -t "ttft_ms<=1500" --json results.json --markdown report.md
noelle eval --recipe nightly.yaml --baseline last-week.json --max-regression 0.1
```

A recipe bundles a run:

```yaml
name: nightly
models: [HuggingFaceTB/SmolLM2-360M-Instruct, local/SmolLM2-360M-Instruct-mlx-4bit]
datasets: [builtin:instruction-following, ./support-tickets.jsonl]
params: {max_new_tokens: 256, temperature: 0}
thresholds: [accuracy>=0.7, tok_s>=25]
```

Dataset files are JSONL: an optional `{"_dataset": {"name", "title", "description"}}` line, then one case per line —
`{"id", "prompt" | "messages", "expected", "scorer": "exact|contains|contains_all|regex|numeric|choice|json|none"}`.

### Using the API from other tools

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8765/v1", api_key="not-needed")
for chunk in client.chat.completions.create(model="HuggingFaceTB/SmolLM2-135M-Instruct",
                                            messages=[{"role": "user", "content": "Hi"}], stream=True):
    print(chunk.choices[0].delta.content or "", end="")
```

The API Server page has copy-ready snippets for curl, Cursor, Continue and LangChain.

### MCP server

`noelle mcp` runs a [Model Context Protocol](https://modelcontextprotocol.io) server on stdio, so coding agents such
as Claude Code or Cursor can use Noelle as a set of tools. To register it with Claude Code:

```bash
claude mcp add noelle -- noelle mcp
```

Other clients take a JSON entry. `noelle mcp --print-config` prints one with the exact command for your install: the
virtualenv's Python for a source checkout, the app executable for a packaged install. If `noelle` is not on your
PATH, run the app executable instead, for example `/Applications/Noelle.app/Contents/MacOS/Noelle mcp --print-config`
on macOS.

| Tool | What it does |
|---|---|
| `hardware_info` | Accelerator, memory budget, memory bandwidth and installed engines |
| `list_local_models` | Models downloaded or built in Noelle (`api_name` is what `chat` and `embed` take) |
| `search_models` | Hub search with a 0–100 fit score for this machine |
| `inspect_model` | What a Hub repository is and how to run it here: run options, recommendation, largest context, speed |
| `chat` | Chat completion with a local model (`"model": "auto"` lets Noelle choose) |
| `embed` | Embedding vectors from a local embedding model |
| `list_knowledge_bases`, `query_knowledge_base` | Search your indexed document collections |

`chat`, `embed` and the knowledge-base tools call Noelle's API server, so start it on the API Server page first. The
MCP server reads the port and API key from Noelle's settings; `--api-url` overrides the address. It only sends
requests to 127.0.0.1, localhost or ::1, so prompts never leave your machine. `search_models` and `inspect_model`
query the Hugging Face Hub. There are no tools to download, load or delete models, and no file or shell access.

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
