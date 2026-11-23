# Product Requirements Document (PRD): Noelle — Local AI Model Lab & OpenAI-Compatible Gateway

**Project Identifier:** `noelle`  
**Live Site URL:** [https://noelle.nevatal.id](https://noelle.nevatal.id)  
**Local Gateway Port:** `http://localhost:10095` (Landing & Docs), `http://localhost:8765` (OpenAI Local API)  
**Ecosystem:** [nevatal.id](https://nevatal.id) &bull; **License:** MIT License  
**Author / Maintainer:** Thirza Ahmad Tsaqif &bull; Contact: `nevatalapp@gmail.com`  

---

## 1. Overview & Problem Statement

### 1.1 Problem Statement
Developers, data scientists, and privacy-conscious professionals increasingly demand local, on-device artificial intelligence. Relying on cloud-hosted closed models (OpenAI, Anthropic, Google Gemini) introduces severe liabilities:
1. **Data Leakage & Compliance Violations:** Proprietary source code, customer records, patient documents, and financial data transmitted over public networks violate GDPR, HIPAA, and strict NDAs.
2. **Recurring API Costs & Rate Limits:** High-volume automated workflows (batch evaluations, agentic loops, unit testing) rack up massive monthly API invoices and suffer from sudden cloud throttling.
3. **Tooling Fragmentation:** The open-source AI landscape is scattered: users must cobble together separate tools for model downloading (`huggingface-cli`), quantization (`llama-quantize`), execution (`llama.cpp` / `ollama`), evaluation (`lm-evaluation-harness`), and IDE integration.
4. **Hardware Guesswork & Memory Overflows:** Selecting the right model size and quantization level (Q4_K_M, Q5_K_S, Q8_0) requires complex VRAM calculations, leading to frequent Out-Of-Memory (OOM) crashes and sluggish CPU swap thrashing.
5. **Absence of Rigorous Local Evaluation:** Developers download quantized models without any quantitative benchmark showing how severely quantization degrades accuracy on their specific industry tasks.

### 1.2 Solution
**Noelle** is an all-in-one, free, open-source desktop AI laboratory and local developer gateway engineered for macOS, Windows, and Linux. Built with **PySide6 (Qt 6)** and a modular native runtime service, Noelle unifies the entire local AI lifecycle:

* **Automated Hardware Profiler & Model Advisor:** Probes physical CPU cores, system RAM, Metal/CUDA/Vulkan GPU VRAM, and context window limits to calculate exact memory headroom, flagging whether a candidate model will run in ultra-fast GPU VRAM or spill into system memory.
* **Isolated Subprocess Runtime:** Inference executes inside a dedicated background worker process isolated with Hugging Face offline mode (`HF_HUB_OFFLINE=1`) and protected via per-session loopback tokens, preventing credential leakage and interface freezing.
* **On-Device Evaluation Lab:** An automated quantitative testing workbench that benchmarks accuracy, token throughput (tokens/sec), prompt evaluation latency, and hallucination rates across customized test datasets.
* **Integrated Quantization Engine:** Converts uncompressed Hugging Face FP16/BF16 weights into optimized GGUF, AWQ, or MLX formats directly from the UI with zero command-line friction.
* **Local OpenAI-Compatible API Server (`:8765`):** Serves `/v1/chat/completions`, `/v1/models`, and `/v1/embeddings` with built-in DNS rebinding protection and localhost loopback constraints.
* **Model Context Protocol (MCP) Server:** Native MCP endpoint allowing modern coding assistants (Cursor, Claude Desktop, Antigravity) to query local models and run on-device evaluations.
* **Local Knowledge & RAG Engine:** Built-in semantic chunking, vector indexing, and hybrid retrieval over local PDFs and codebases with zero external data transmissions.
* **Nevatal Golden Yellow Brand Identity:** Sleek, high-contrast visual design system featuring Nevatal's signature `#FFC700` cyber gold with OLED Obsidian `#0B0B0D` and neutral surfaces.

---

## 2. Goals & Non-Goals

### 2.1 Goals
* **100% Data Sovereignty:** Guarantee that prompts, documents, embeddings, and completions never touch external networks or telemetry servers.
* **Zero-Setup Model Discovery:** Direct search and one-click download from Hugging Face with automatic SHA-256 checksum verification and token-gated repository support.
* **Automated Hardware Matching:** Categorize models into "Runs entirely in VRAM (Fast)", "Spills into RAM (Moderate)", or "Too Large" before the user downloads a single byte.
* **Drop-in OpenAI Replacement:** Seamless interoperability with standard `openai` Python/Node SDKs, LangChain, LlamaIndex, and AutoGen via standard REST endpoints.
* **Reliable Subprocess Isolation:** Completely decouple the GUI thread from heavy C++ matrix operations to prevent app crashes and memory leaks.
* **Native OS Secret Management:** Store sensitive Hugging Face user tokens exclusively in the host OS Keychain, Windows Credential Manager, or Linux Secret Service.

### 2.2 Non-Goals
* **Not a Cloud Hosting Provider:** Noelle does not provide cloud-hosted GPU instances or relay traffic through proxy servers.
* **Not an Online Multi-User SaaS:** Designed strictly as a single-tenant on-device workstation utility.
* **Not an Automated Web Scraper:** Does not harvest internet data without explicit user request (optional SearXNG query handoffs only).

---

## 3. Product Features & User Experience

```
┌────────────────────────────────────────────────────────────────────────┐
│                      Noelle Desktop Application                        │
├────────────────────────────────────────────────────────────────────────┤
│ Top Navigation: [⚡ Mission] [🔍 Discover] [📊 Evaluate] [📚 Knowledge]│
│                 [💾 Storage] [⚙ Settings]               [● Local API]  │
├───────────────────────────────────┬────────────────────────────────────┤
│ Left Control & Inspector Panel    │ Active Workspace Viewport          │
│ ┌───────────────────────────────┐ │ ┌────────────────────────────────┐ │
│ │ Hardware Telemetry:           │ │ │ Mission Mode (Interactive)     │ │
│ │ • CPU: 16 Cores | RAM: 32 GB  │ │ │ Prompt / Generation Studio     │ │
│ │ • GPU: Apple M-Series (Metal) │ │ ├────────────────────────────────┤ │
│ │ • VRAM Headroom: 24.2 GB Free │ │ │ Evaluation Lab Workbench       │ │
│ ├───────────────────────────────┤ │ │ • Task: Summarization Accuracy │ │
│ │ Active Model:                 │ │ │ • Accuracy: 94.2% | 48.6 tok/s │ │
│ │ • Llama-3.2-3B-Instruct.Q4_K_M│ │ ├────────────────────────────────┤ │
│ │ • Context Window: 8,192 tok   │ │ │ Knowledge Base (Local RAG)     │ │
│ │ • Quantization: 4-bit Medium  │ │ │ • 14 PDFs Indexed (Local DB)   │ │
│ └───────────────────────────────┘ │ └────────────────────────────────┘ │
└───────────────────────────────────┴────────────────────────────────────┘
```

### 3.1 Mission Mode (Prompt Studio & Chat)
* Structured prompt crafting for writing, code generation, summarization, and data extraction.
* Streaming responses with live tokens-per-second counters, time-to-first-token (TTFT) metrics, and memory gauges.
* Hot-swappable LoRA adapter insertion without model re-initialization.

### 3.2 Discover & Hardware Advisor
* Direct exploration of trending Hugging Face repositories with automatic architecture detection (Llama, Mistral, Qwen, DeepSeek, Phi).
* Dynamic quantization selector showing disk footprint, estimated RAM/VRAM load, and expected generation speed.
* Safe sandbox warning and quarantine for repositories containing executable Python scripts (`trust_remote_code=True`).

### 3.3 Evaluation Lab
* Quantitative testing suite running reproducible benchmarks against customized CSV/JSONL datasets.
* Comparative side-by-side model matrix analyzing accuracy scores, perplexity, latency, and hallucination rates across different quantization levels.
* Exportable Markdown and JSON evaluation reports for CI/CD regression validation.

### 3.4 Local Knowledge & RAG
* On-device document ingestion supporting PDF, Markdown, Plain Text, and Source Code.
* Local chunking and embeddings generated using private on-device embedding models.
* Hybrid keyword and dense vector retrieval with cosine similarity filtering.

### 3.5 Developer Gateway (Local API & MCP)
* OpenAI-compatible HTTP server running on `127.0.0.1:8765`.
* Model Context Protocol (MCP) server supporting IDE coding agents (Cursor, Claude Desktop, Antigravity) with tools: `list_models`, `ask_model`, and `run_eval`.
* Strict security defenses against DNS rebinding and cross-site simple POST requests.

---

## 4. Key Performance Indicators (KPIs)

* **Zero Memory Leaks:** Subprocess memory reclaimed completely on model unload.
* **Sub-20ms API Overhead:** Local REST routing latency under 20ms excluding model inference time.
* **100% Offline Integrity:** Zero outbound network packets during active model inference and knowledge retrieval.
* **High Contrast Accessibility:** Light and dark palettes strictly meeting WCAG 2.1 AA standards with Nevatal yellow `#FFC700` branding.
