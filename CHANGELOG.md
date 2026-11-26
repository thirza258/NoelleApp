# Changelog

All notable changes to the Noelle Desktop AI Studio, CLI, local API, and ecosystem will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Online documentation and release notes are also available at [https://noelle.nevatal.id/docs/changelog/](https://noelle.nevatal.id/docs/changelog/).

---

## [1.2.0] - 2026-11-26

### Added
- **Model Runnability Rules**: Pure-Python runnability evaluation rules (`noelle/runtime/runnability.py`) executed across both desktop shell and worker daemon.
- **Worker Capability Advertising**: Worker `HELLO` handshake advertises `RUNTIME_CAPS` to keep UI capability detection in sync with installed accelerators and packages.
- **Discover "Runs in Noelle" Filter**: Instant filter on the Discover page to display only checkpoints verified to run on the host system.
- **Runnability Verdict Badges**: Detailed badges on model cards and headers indicating native execution, 4-bit requirements, CPU fallbacks, or unsupported architectures.
- **Typed Load Failure Codes**: Failures report explicit codes (`UNSUPPORTED`, `NEEDS_NEWER_RUNTIME`, `GATED`) rather than raw library errors, with local filesystem cache paths sanitized and scrubbed.
- **Gated Model Access**: Preflight authentication check via `HfApi.auth_check` before initiating downloads; terminal `GATED` status with direct "Open model page" action to request access or accept terms on Hugging Face.
- **UI Error Line**: Shared `ErrorLine` component that renders escaped, wrapped, selectable, and copyable error text with an immediate "Open logs" button.
- **UI Lifecycle Guard**: Toast notification lifecycle guards and bound download slots preventing UI crashes during rapid tab navigation.
- **Diagnostics Service**: New `noelle/app/services/diagnostics.py` service to package debug reports with device telemetry and log excerpts.
- **Agent Runtime Foundation**: Initial modules for autonomous agent runtime including `repository_analyzer.py`, `environment_planner.py`, and runtime adapter factory (`adapters/factory.py`).
- **v1.2 Core Services**: SQLite conversation store (`conversation_store.py`), privacy-preserving update checker (`update_checker.py`), external Ollama and LM Studio model discovery (`external_models.py`), multi-provider manager (`provider_manager.py`), cost ledger (`cost_ledger.py`), and intelligent model router (`model_router.py`).
- **Test Fixtures**: Checkpoint fixture matrix covering ASR, Diffusion, Embeddings, LLM text, Kokoro/OmniVoice TTS, and Vision model families.

---

## [1.1.1] - 2026-11-25

### Added
- **CUDA 13.2 Wheels**: Windows installers updated with PyTorch CUDA 13.2 wheels (`cu132`) for CPython 3.11–3.14.

### Changed
- **High-Precision Windows Telemetry**: Replaced `time.monotonic()` (~15.6 ms resolution) with `time.perf_counter()` for microsecond workflow node profiling on Windows.
- **Slim Linux Assets**: Maintained CPU-optimized packages on Linux to keep installer bundles under 2 GB.

---

## [1.1.0] - 2026-11-25

### Added
- **Hybrid Cloud + Local AI Workflow Studio**: Visual node-based workflow editor (`noelle/workflow/`) supporting LLM, Embedding, STT, TTS, Vision, and RAG pipelines.
- **Cycle Detection & Zero Egress**: Graph validation with cycle detection and strict Local-Only Zero Egress enforcement to prevent accidental network egress.
- **Unified Provider Engine**: Seamlessly switch between Local inference, OpenAI, and OpenRouter backends (`noelle/providers/`).
- **Real-Time Telemetry**: Real-time marginal cost calculation, VRAM profiling, and token pricing with OS keyring secret management.
- **Cost Accounting & Decision Advisor**: Built-in financial advisor (`noelle/services/cost_advisor.py`) calculating query break-even points, TCO, and capital expenditure ROI vs cloud APIs.
- **Adversarial Evaluation Labs**: Specialized evaluation generators for Indonesian currency/jargon TTS synthesis, STT Levenshtein WER/CER, and dense embedding retrieval benchmarks (`noelle/evaluation/generators/`), plus Audio A/B Lab and Embedding Lab.
- **CLI Workflow Tools**: Added `noelle workflow run`, `noelle workflow test-gen`, and `noelle workflow advisor` commands.

### Fixed
- **Atomic Settings Persistence**: Fsynced atomic configuration writes to eliminate settings truncation on unexpected shutdowns.
- **IPC Safety**: Bounded ZeroMQ message payloads to guard against buffer overruns.
- **Process Management**: Graceful teardown of child download processes.
- **Exception Logging**: Global logging for uncaught Qt/PySide6 exceptions.

---

## [1.0.1] - 2026-11-24

### Added
- **Documentation Web Portal**: 14 guides published on `noelle.nevatal.id` covering quickstart, installation, evaluation, RAG, API, MCP, and troubleshooting.
- **SourceForge Distribution**: Mirrors and automated installer links configured for Windows, macOS, and Linux.
- **AI Agent Discovery**: Added `.well-known/agent-card.json`, `.well-known/ai-catalog.json`, `api-catalog`, `llms.txt`, and WebMCP definitions.
- **Docker Deployment**: Containerized deployment configuration for the landing site and documentation with Nginx security rules.

### Security
- **DNS Rebinding Protection**: Local API server enforces `HostGuardMiddleware`, restricting requests to loopback hosts (`127.0.0.1`, `localhost`, `[::1]`).
- **Strict Content-Type**: Required `application/json` for API POST endpoints to force browser CORS preflight and block cross-site simple POST requests.

### Fixed
- **Windows Launcher**: Resolved UTF-8 / Unicode console encoding issues in `launcher.py` and `noelle.spec`.
- **Packaging Dependencies**: Standardized on CUDA 12.4 LTS for installers.

---

## [1.0.0] - 2026-11-23

### Added
- **PySide6 Native Desktop Shell**: Modern desktop interface with light, dark, and system themes. UI never imports `torch` directly.
- **Isolated ZeroMQ Worker**: Out-of-process inference daemon with single GPU lane, VRAM pre-flight checks, LRU cache eviction, and OOM isolation.
- **Hub Discover Browser**: Instant compatibility scoring (0–100 scale), modality filters, and search.
- **Model Workspace & Analysis**: Compatibility breakdown (memory, runtime, precision, context, trust) and runtime profiles (Balanced, Quality, Fast, Low memory, Max context).
- **Multi-Modal Playgrounds**: Built-in test playgrounds for Chat, Vision Q&A, Object Detection, Speech-to-Text (ASR), Speech Synthesis (TTS), Text-to-Image Diffusion, and Embeddings.
- **RAG Studio (Knowledge Canvas)**: Local document ingestion (PDF, DOCX, Markdown, code), vector embeddings, and `sqlite-vec` semantic search with retrieval accuracy evaluation.
- **Mission Mode**: High-level planner converting natural language goals into local AI stack recommendations.
- **Evaluation Lab**: 8 built-in datasets, deterministic local scoring, historical tracking, and regression gating via `noelle eval`.
- **Integrated llama.cpp / GGUF**: Automated binary downloading with SHA-256 verification and per-session random auth keys for `llama-server`.
- **Local OpenAI API Server**: OpenAI-compatible `/v1/chat/completions`, `/v1/embeddings`, `/v1/audio/transcriptions`, `/v1/images/generations`, and `/v1/models` at `http://127.0.0.1:8765/v1`.
- **Model Context Protocol (MCP)**: Native stdio MCP server (`noelle mcp`) for Claude Code, Cursor, and agent integrations.
- **Platform Packaging**: macOS `.pkg`/`.dmg`, Windows Inno Setup `.exe`, and Linux `.deb` installers.
