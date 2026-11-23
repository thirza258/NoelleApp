# Architecture Document: Noelle — Local AI Model Lab & OpenAI-Compatible Gateway

## 1. System Topology & Process Isolation

Noelle enforces a strict **dual-process architecture** designed to eliminate GUI thread blocking, isolate volatile native C++ libraries, and maintain an impenetrable security boundary:

```
┌────────────────────────────────────────────────────────────────────────┐
│                   Desktop User Interface Process                       │
│                        (PySide6 / Qt 6 GUI)                            │
│  ┌───────────────────────┐ ┌──────────────────────┐ ┌───────────────┐  │
│  │ Mission & Chat Studio │ │ Discover & Advisor   │ │ Knowledge RAG │  │
│  └───────────────────────┘ └──────────────────────┘ └───────────────┘  │
│  ┌───────────────────────┐ ┌──────────────────────┐ ┌───────────────┐  │
│  │ Evaluation Lab UI     │ │ Quantization Studio  │ │ Theme Engine  │  │
│  │                       │ │                      │ │ (#FFC700 Gold)│  │
│  └───────────────────────┘ └──────────────────────┘ └───────────────┘  │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    │ Local IPC Protocol (TCP Loopback)
                                    │ Auth: Ephemeral Per-Session Token
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                   Worker Runtime Subprocess (Isolated)                 │
│                 (Hugging Face Offline: HF_HUB_OFFLINE=1)               │
│                                                                        │
│  ┌─────────────────────────┐          ┌─────────────────────────────┐  │
│  │ Model Lifecycle Manager │          │ Hardware Acceleration       │  │
│  │ • GGUF Context Allocator│◄────────►│ • Apple Silicon Metal       │  │
│  │ • LoRA Hot-Swap Queue   │          │ • NVIDIA CUDA Runtime       │  │
│  │ • Quantization Engine   │          │ • Vulkan / CPU Fallback     │  │
│  └────────────┬────────────┘          └─────────────────────────────┘  │
│               │                                                        │
│               ▼                                                        │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ Local Services & Gateway Tier                                    │  │
│  │ • OpenAI-Compatible REST Server (:8765)                           │  │
│  │   - DNS Rebinding Protection (Host Header Verification)          │  │
│  │   - Strict application/json Enforcement                          │  │
│  │ • Model Context Protocol (MCP) Server (JSON-RPC stdio)            │  │
│  │ • Local Vector RAG Engine & Embedding Indexer                     │  │
│  │ • Evaluation Benchmark Runner & Telemetry Collector              │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                   ┌────────────────┴────────────────┐
                   ▼                                 ▼
      ┌─────────────────────────┐       ┌─────────────────────────┐
      │  Local Model Cache      │       │  OS Credential Vault    │
      │  (~/.cache/noelle/)     │       │  • macOS Keychain       │
      │  • GGUF Weights         │       │  • Windows Cred Manager │
      │  • Vector DB Embeddings │       │  • Linux Secret Service │
      │  • Benchmark JSONL Logs │       │  (Zero Plaintext Leaks) │
      └─────────────────────────┘       └─────────────────────────┘
```

---

## 2. Component Breakdown

### 2.1 The Application Presentation Layer (`noelle.app`)
* **Framework:** PySide6 (Qt 6.8+ bindings for Python).
* **Theme & Design System (`noelle.app.theme`):**
  * Implements the **Nevatal Yellow Brand Identity** using `--accent: #FFC700` and `--on-accent: #0B0B0D`.
  * Dual palette system: Pure Light mode and OLED Neutral Grey Dark mode (`#1E1E1E` background with `#232323` surfaces).
  * High-contrast styling for code snippets, markdown prose, benchmark charts, and status indicators.
* **Asynchronous Threading:** All long-running disk operations, network downloads, and quantization jobs run in background Qt worker threads (`QThread` / `QThreadPool`), communicating via signal-slot mechanisms to keep the 60 FPS UI smooth.

### 2.2 The Runtime & Isolation Subprocess (`noelle.runtime`)
* **Process Separation:** Launched via Python `subprocess.Popen` as `noelle.runtime.worker_service`.
* **Zero Credential Exposure:** The worker process **never receives the user's Hugging Face API key**. Model files are downloaded and verified in the UI/downloader thread; the worker is only handed verified local filesystem paths.
* **Network Isolation:** Hardened with `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`, rendering the runtime completely unable to make accidental outbound network requests.
* **IPC Protocol:** Low-latency JSON messaging over loopback TCP (`127.0.0.1`). Authentication is established via an ephemeral 256-bit cryptographically random token passed once over stdin on startup.

### 2.3 Hardware Intelligence & Compute Engine (`noelle.runtime.devices` & `advisor`)
* **Dynamic Hardware Detection:**
  * **macOS:** Queries Metal framework and unified memory architecture (UMA) via `sysctl` and CoreGraphics APIs.
  * **Windows & Linux:** Probes `nvidia-smi` / NVML for CUDA-capable GPUs, ROCm for AMD Radeon GPUs, and fallback to Vulkan or CPU AVX-512 / AVX2 threads.
* **VRAM Overhead Calculator:** Evaluates candidate model parameter count ($P$), quantization bit-width ($B$), and context length ($C$):
  $$\text{Memory Required} \approx \frac{P \times B}{8} + \left(2 \times L \times H \times C \times \text{bytes per KV}\right) + \text{Runtime Overhead}$$
  Models are flagged as "Full VRAM (Fast)", "Partial Offload", or "CPU Only" with millisecond precision.

### 2.4 Model Quantization Workbench (`noelle.runtime.quantization_engine`)
* **Integrated Formats:** Supports direct quantization to GGUF (Q4_K_M, Q5_K_M, Q8_0, etc.) using native `llama.cpp` tools.
* **Apple Silicon MLX:** Leverages native Apple MLX 4-bit and 8-bit weight compression on macOS.
* **Integrity Validation:** SHA-256 hash checks performed automatically before and after quantization passes.

### 2.5 Local OpenAI-Compatible REST Server (`noelle.runtime.server_service`)
* **Port:** Default `:8765` bound to `127.0.0.1` and `::1`.
* **Endpoints:**
  * `GET /v1/models` — Lists active and cached local models.
  * `POST /v1/chat/completions` — Streaming (SSE) and non-streaming OpenAI chat completion format.
  * `POST /v1/embeddings` — On-device vector representations for text.
* **Web Threat Mitigation:**
  * **Anti-DNS Rebinding:** Strict `Host` header enforcement rejecting requests not directed to `localhost`, `127.0.0.1`, or `[::1]`.
  * **Anti-CSRF:** Enforces `Content-Type: application/json` on all mutable endpoints, blocking simple cross-site HTML form submissions from malicious websites.
  * **Bearer Authentication:** Optional local API token configurable via UI settings.

### 2.6 Model Context Protocol (MCP) Server (`noelle.runtime.mcp_server`)
* **Standard:** Model Context Protocol specification over stdio.
* **Exposed Tools:**
  * `list_models()`: Enumerate downloaded weights and active status.
  * `ask_model(prompt, model, temperature)`: Query local models directly from IDE extensions (Cursor, Claude Desktop, Antigravity).
  * `benchmark_model(model, dataset)`: Trigger on-device evaluation runs and stream back performance telemetry.

### 2.7 Local Knowledge Base & RAG Pipeline (`noelle.runtime.rag_engine`)
* **Parsing:** Native extraction from PDF, Markdown, and source code.
* **Chunking:** Semantic sliding-window chunking with configurable overlap.
* **Storage:** Embedded vector database residing in `~/.cache/noelle/vectors.db`.
* **Search:** Cosine similarity calculation over local embeddings with threshold filtering.

---

## 3. Web & Landing Architecture (Port 10095)

```
                     ┌────────────────────────────────────┐
                     │          User Web Browser          │
                     └─────────────────┬──────────────────┘
                                       │ Port 10095 (HTTP)
                     ┌─────────────────▼──────────────────┐
                     │       Nginx Alpine Container       │
                     │          (noelle-landing)          │
                     │  • Port Mapping: 10095:80          │
                     │  • Gzip Compression Enabled        │
                     │  • Custom 404 Fallback Routing     │
                     └─────────────────┬──────────────────┘
                                       │
        ┌──────────────────────────────┼──────────────────────────────┐
        ▼                              ▼                              ▼
┌──────────────────┐          ┌──────────────────┐          ┌──────────────────┐
│  Landing Page    │          │  Documentation   │          │  Compliance      │
│  • index.html    │          │  • /docs/        │          │  • /docs/privacy/│
│  • styles.css    │          │  • /docs/api/    │          │  • /docs/terms/  │
│  • main.js       │          │  • /docs/quick…  │          │  • /privacy 301  │
│  • assets/       │          │  • /docs/mcp/    │          │  • /terms 301    │
└──────────────────┘          └──────────────────┘          └──────────────────┘
```

* **Static Site Generator (`landing/build.py`):** Compiles documentation sources (`landing/docs-src/*.md`) into zero-framework, static HTML pages, synchronizing the site map, robots.txt, 404.html, and footer links.
* **Containerized Deployment:** Packaged via `Dockerfile` and `docker-compose.yml`, deployed on an isolated, non-standard port **`10095`**, matching the production server conventions of `nevatal.id`.
* **Zero Analytics & Pure Performance:** Zero external CDN dependencies; all CSS, fonts, and JS are bundled locally, loading in under 15ms.
