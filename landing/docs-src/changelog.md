---
title: Changelog
nav: Changelog
description: Complete release history, new capabilities, and improvements across every version of Noelle.
section: Reference
order: 32
slug: changelog
---

All notable changes to the Noelle Desktop AI Studio, CLI, local API, and ecosystem are documented here.
Versions adhere to [Semantic Versioning](https://semver.org/).

---

## Version 1.2.0 — November 26, 2026

Version 1.2.0 introduces runtime runnability preflight checks, gated model access flows, resilient UI error handling, the foundation for autonomous agent workflows, and expanded local service backends.

### Model runnability & compatibility preflight
* **Shared pure-Python runnability rules**: Replaced tag-only heuristics with a unified evaluation engine (`noelle/runtime/runnability.py`) executed across both the desktop UI and the out-of-process worker daemon.
* **Worker capability advertising**: The worker `HELLO` handshake now publishes dynamic `RUNTIME_CAPS`, ensuring the UI accurately reflects installed accelerators and libraries.
* **Discover "Runs in Noelle" filter**: Added an instant toggle on the Discover page to hide models that cannot execute on the current machine.
* **Runnability verdict badges**: Every model card and detail header displays a crystal-clear verdict (`Runs natively`, `Needs 4-bit build`, `CPU fallback`, `Requires newer runtime`, or `Unsupported architecture`).
* **Typed load failure codes**: Failures now report clean codes (`UNSUPPORTED`, `NEEDS_NEWER_RUNTIME`, `GATED`) rather than raw library stack traces, with all local filesystem cache paths sanitized and scrubbed.

### Gated model access & download flow
* **Preflight authentication check**: Download requests verify access against `HfApi.auth_check` before starting multi-gigabyte file transfers.
* **Terminal GATED state**: Replaced generic retry loops with a dedicated `GATED` status and an immediate **Open model page** action to accept license agreements or request repository permissions on Hugging Face.

### UI diagnostics & error resilience
* **Shared `ErrorLine` component**: Model errors are presented in a selectable, copyable, and wrapped container with an integrated **Open logs** button.
* **Toast notification lifecycle guard**: Bound download slots and state timers to prevent race conditions and crashes during rapid UI tab switching.
* **Diagnostics bundle exporter**: Built-in diagnostics service (`noelle/app/services/diagnostics.py`) to gather anonymized runtime status, device telemetry, and log excerpts for troubleshooting.

### Autonomous agent runtime foundation
* **Repository analyzer**: Added `noelle/runtime/agent/repository_analyzer.py` for automated code inspection, dependency discovery, and agent task planning.
* **Environment planner**: Built `noelle/runtime/agent/environment_planner.py` featuring automated environment verification and resolution loops.
* **Runtime adapter factory**: Introduced modular adapter interfaces (`noelle/runtime/adapters/`) for extensible agent execution.

### v1.2 Core services & multi-model routing
* **Conversation store**: Multi-turn conversation persistence powered by SQLite (`noelle/app/services/conversation_store.py`).
* **Background update checker**: Privacy-preserving update notifications for the desktop application and downloaded Hub models (`noelle/app/services/update_checker.py`), strictly throttled to 24-hour intervals and disabled when local-only mode is active.
* **External model discovery**: Scan and register local models from Ollama and LM Studio (`noelle/app/services/external_models.py`).
* **Provider management & cost ledger**: Track query costs and token quotas across local and remote backends (`noelle/app/services/provider_manager.py`, `noelle/app/services/cost_ledger.py`).
* **Intelligent model router**: Route incoming tasks automatically to the best available model based on capability and cost (`noelle/app/services/model_router.py`).

### Verification & tests
* Comprehensive test fixture matrix covering all supported checkpoint families: ASR (CTranslate2, NeMo), Diffusion (Single-file, SD), Embeddings (Model2Vec, Sentence-Transformers), Text (GGUF, ONNX, Transformers), TTS (Kokoro, OmniVoice, Stable Audio 3), and Vision/VLM.
* End-to-end and unit regression suites for desktop UI lifecycle stability.

---

## Version 1.1.1 — November 25, 2026

Version 1.1.1 brings PyTorch CUDA 13.2 support for Windows and high-precision telemetry for low-latency nodes.

### Performance & hardware builds
* **CUDA 13.2 for Windows**: Updated Windows standalone installer packages with PyTorch CUDA 13.2 wheels (`cu132`) for CPython 3.11–3.14.
* **Optimized Linux footprint**: Maintained a lightweight, CPU-optimized distribution for Linux releases to keep downloads under 2 GB.
* **High-precision Windows telemetry**: Switched node latency measurement on Windows from `time.monotonic()` (~15.6 ms resolution) to `time.perf_counter()` for microsecond profiling accuracy.

---

## Version 1.1.0 — November 25, 2026

Version 1.1.0 adds the Hybrid Cloud + Local AI Workflow Studio, comprehensive financial cost modeling, and adversarial model evaluation labs.

### Hybrid Cloud + Local AI Workflow Studio
* **Visual Workflow Canvas**: Node-based graphical workspace (`noelle/workflow/`) allowing users to assemble multi-stage pipelines combining LLM, Embedding, Speech-to-Text, Speech Synthesis, Vision, and RAG nodes.
* **Cycle detection & graph validation**: Strict topological execution graph with instant cycle detection.
* **Zero Egress Local-Only enforcement**: Built-in privacy gate guaranteeing that sensitive local pipelines never transmit data or tokens across external network boundaries.
* **Unified Provider Engine**: Seamlessly switch or chain nodes between Local models, OpenAI, and OpenRouter backends (`noelle/providers/`).
* **Real-time telemetry**: Per-node and end-to-end latency tracking, VRAM allocation profiling, and token pricing calculated using live provider tariffs.

### Financial Decision Advisor & cost accounting
* **Cost Advisor**: Built-in financial analysis engine (`noelle/services/cost_advisor.py`) that computes break-even query thresholds, Total Cost of Ownership (TCO), and capital expenditure payback periods comparing local hardware investments against cloud API tariffs.

### Adversarial Evaluation Labs
* **Adversarial test generators**: Built generators for demanding evaluation domains (`noelle/evaluation/generators/`), including Indonesian currency and technical jargon TTS synthesis, STT Levenshtein Word/Character Error Rate (WER/CER) scoring, and dense embedding retrieval benchmarks.
* **Domain labs**: Added specialized Audio A/B Lab (`audio_ab_lab.py`) and Embedding Lab (`embedding_lab.py`).

### CLI workflow tools
* Added command-line interfaces for headless automation:
  * `noelle workflow run`: Execute workflow pipelines headlessly in CI or scripts.
  * `noelle workflow test-gen`: Synthesize reproducible adversarial test datasets.
  * `noelle workflow advisor`: Run financial break-even analysis from the command line.

### System hardening
* **Atomic settings persistence**: Implemented fsynced atomic configuration writes to eliminate corruption during unexpected shutdowns.
* **Bounded IPC payloads**: Safeguarded ZeroMQ IPC communications against oversized message buffer overruns.
* **Graceful child process termination**: Clean teardown of download and extraction subprocesses.
* **Qt unhandled exception logging**: Centralized error logging for uncaught PySide6 exceptions.

---

## Version 1.0.1 — November 23–24, 2026

Version 1.0.1 launched the documentation portal, official distribution channels, security hardening, and agent discovery metadata.

### Documentation & web portal
* **Official documentation**: Published 14 comprehensive guides on `noelle.nevatal.id` covering setup, quantization, evaluation, RAG, API, MCP, and troubleshooting.
* **Automated site generator**: Custom static site generator (`landing/build.py`) with automatic sitemap generation, theme synchronization, and navigation compilation.
* **SourceForge distribution**: Configured official distribution mirrors on SourceForge for Windows `.exe`, macOS `.pkg`/`.dmg`, and Linux `.deb` installers.

### AI agent discovery & standards
* **Agent Cards & catalogs**: Added RFC-compliant discovery metadata including `.well-known/agent-card.json`, `ai-catalog.json`, `api-catalog`, and OAuth authorization server manifests.
* **LLM context overview**: Added machine-readable architecture and usage specification via `llms.txt`.
* **WebMCP integration**: Web-accessible MCP tool definitions for browser-based agents.

### Security & platform packaging
* **Local API security hardening**: Added `HostGuardMiddleware` blocking DNS rebinding attacks by restricting `Host` headers to loopback addresses (`127.0.0.1`, `localhost`, `[::1]`). Required strict `application/json` content-type to block cross-site simple POST requests.
* **Windows Unicode fix**: Resolved UTF-8 command-line encoding issues in the Windows PyInstaller launcher.
* **Ecosystem branding**: Applied the Nevatal yellow theme palette across the desktop app and documentation.

---

## Version 1.0.0 — November 23, 2026

Initial major release of the Noelle Desktop AI Studio, transforming the project into a local-first workbench for open AI models.

### Desktop AI Studio architecture
* **PySide6 Native Desktop Shell**: Modern desktop interface with light, dark, and system-adaptive themes. The UI process never imports heavy machine learning frameworks, ensuring responsiveness.
* **Isolated ZeroMQ worker**: Dedicated worker daemon communicating over loopback IPC with session-token authentication. Features single GPU-lane queuing, VRAM pre-flight checks, LRU cache eviction, and out-of-memory containment.
* **Multi-engine execution**: Native acceleration across Hugging Face Transformers (CUDA, MPS, CPU), llama.cpp (GGUF), MLX (Apple Silicon Metal), Diffusers, and Sentence-Transformers.

### Core studios & features
* **Hub Discover**: Model browser with instant hardware compatibility scoring (0–100 scale), modality filters, and search.
* **Model Workspace & Analysis**: Granular breakdown of memory fit, precision tradeoffs (FP16, INT8, NF4, AWQ, GPTQ, GGUF), context window limits, and trust controls for repositories shipping custom Python code.
* **Interactive Playgrounds**: Built-in test benches for Chat, Vision Q&A, Object Detection, Speech-to-Text (ASR), Speech Synthesis (TTS), Text-to-Image Diffusion, and Embeddings.
* **RAG Studio (Knowledge Canvas)**: Local document ingestion (PDF, DOCX, Markdown, code), vector embeddings, and `sqlite-vec` semantic search with retrieval accuracy evaluation (hit rate, MRR, nDCG@k).
* **Mission Mode**: High-level planner that converts natural language project requirements into tailored local AI hardware and model recommendations.
* **Evaluation Lab**: 8 built-in benchmark datasets, deterministic local scoring, historical experiment tracking, and regression gating via `noelle eval`.
* **llama.cpp integration**: Integrated GGUF quantization and inference with automated binary verification and SHA-256 integrity checks.
* **Local OpenAI API Server**: Drop-in compatible `/v1/chat/completions`, `/v1/embeddings`, `/v1/audio/transcriptions`, `/v1/images/generations`, and `/v1/models` endpoints listening at `http://127.0.0.1:8765/v1`.
* **Model Context Protocol (MCP)**: Native stdio MCP server (`noelle mcp`) for Claude Code, Cursor, and developer tool integration.
