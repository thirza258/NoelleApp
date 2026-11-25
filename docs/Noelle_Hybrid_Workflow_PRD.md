# Product Requirements Document: Hybrid Cloud + Local AI Workflow & Evaluation Lab

**Project Identifier:** `noelle-hybrid-lab`  
**Parent Product:** Noelle Desktop AI Studio  
**Document Version:** 1.0.0  
**Target Platform:** Desktop / PySide6 (macOS, Windows, Linux)  
**Primary Purpose:** Test, compare, benchmark, and compose cloud AI APIs and locally running models across LLM, embedding, vision, STT, TTS, multimodal, and RAG workloads.

---

## 1. Product Vision, System Positioning, and Strategic Rationale

Modern artificial intelligence development is characterized by an operational divide between hosted cloud application programming interfaces (APIs) and local on-device neural runtimes. Developers, enterprise architects, and machine learning researchers frequently grapple with multi-dimensional trade-offs spanning inference latency, token expenditure, privacy guarantees, and output quality. While managed cloud platforms offer high-parameter reasoning models and elastic throughput, they introduce recurring operational costs, variable network latency, and data governance concerns. Conversely, local execution using open-weight architectures provides absolute data sovereignty, zero marginal API expenses, and predictable offline processing, but operates within rigid local hardware bounds such as GPU video memory (VRAM) and compute throughput.

The **Hybrid Cloud + Local AI Workflow & Evaluation Lab** introduces a foundational architectural capability to the Noelle desktop environment. Rather than positioning the application purely as an isolated local model launcher or a simple chat interface, this subsystem establishes an empirical experimentation laboratory and execution orchestration engine. Within this environment, remote cloud APIs and local open-weight models function as completely interchangeable, interoperable workflow components across all supported modalities, including large language models, text embeddings, computer vision, speech-to-text transcription, text-to-speech synthesis, and retrieval-augmented generation pipelines.

The core product thesis asserts that developers require repeatable, empirical evaluation under identical input conditions to make informed deployment decisions. By abstracting the execution boundary behind standardized workflow nodes, Noelle enables practitioners to systematically benchmark whether an on-device model—such as a quantized Whisper speech recognizer, a local BGE-M3 embedding generator, or an 8B parameter causal language model—satisfies application requirements compared to proprietary hosted endpoints.

```
                         NOELLE
                           │
              ┌────────────┴────────────┐
              │                         │
        Model Discovery            Workflow Lab
              │                         │
              ▼                         ▼
       Runtime Builder             Node Engine
              │                         │
              ▼              ┌──────────┼──────────┐
       Local Runtime          │          │          │
              │             LLM       Audio      Vision
              │              │          │          │
              │          ┌───┴───┐  ┌──┴──┐   ┌──┴──┐
              │          │Cloud  │  │Cloud │   │Cloud│
              │          │Local  │  │Local │   │Local│
              │          └───────┘  └──────┘   └─────┘
              │
              ▼
        Runtime Manifest
              │
              └──────────────────┐
                                 ▼
                         Evaluation Engine
                                 │
              ┌──────────────────┼──────────────────┐
              ▼                  ▼                  ▼
          Accuracy             Latency             Cost
              │                  │                  │
              └──────────────────┼──────────────────┘
                                 ▼
                          Experiment Report
```

---

## 2. Core Abstractions: The Workflow Canvas and Unified Provider Engine

The operational foundation of the Hybrid Lab rests upon two interconnected structural abstractions: the visual **Workflow Graph** and the **Unified Provider Interface**. These systems decouple high-level pipeline composition from low-level execution mechanics.

### 2.1 The Workflow Canvas Topology

The central visual and logical construct within the desktop interface is the Workflow Graph. Users construct directed processing pipelines by assembling functional nodes across multiple modalities. Each node represents a distinct computational operation—such as transcribing raw audio, computing dense semantic vectors, querying a vector index, executing generative inference, or rendering speech.

Pipeline execution flows through sequential and branching stages across user-defined steps:

```
                  ┌───────────────┐
                  │ Test Dataset  │
                  └───────┬───────┘
                          │
                          ▼
                  ┌───────────────┐
                  │     STT       │
                  │ Local Whisper │
                  └───────┬───────┘
                          │
                          ▼
                  ┌───────────────┐
                  │      LLM      │
                  │ OpenRouter    │
                  │ Qwen / Claude │
                  └───────┬───────┘
                          │
                          ▼
                  ┌───────────────┐
                  │     TTS       │
                  │ Local Model   │
                  └───────┬───────┘
                          │
                          ▼
                       Audio
```

1. **Input Signal Ingestion:** An incoming audio file, user prompt, document batch, or image capture enters the pipeline root.
2. **First-Stage Processing:** The input signal is converted into an intermediate representation, such as an audio file routed into an automated speech recognition node.
3. **Core Generative Routing:** The intermediate output passes into a causal language model node or multimodal vision node for reasoning and synthesis.
4. **Context Augmentation Branch:** If grounded retrieval is required, prompt tokens query an embedding node and vector database concurrently, injecting relevant context passages into the language model.
5. **Final Output Synthesis:** Generated text is delivered to the user interface, passed to an external application, or converted into an acoustic waveform via a speech synthesis node.

The decisive capability of this graph is that every individual node exposes an **execution target toggle**. Within a single processing sequence, a user can configure an initial speech-to-text stage to execute locally via on-device Whisper models, route the resulting transcript into a remote cloud language model for multi-step reasoning, pass the output to an on-device embedding model for local vector indexing, and finally trigger a local speech synthesizer for audio playback. Changing an execution target from a cloud endpoint to a local runtime requires altering a single node parameter without reconfiguring the surrounding workflow pipeline.

### 2.2 The Unified Provider Interface

To maintain interoperability across heterogeneous cloud APIs and local runtime engines, the architecture enforces a normalized provider interface. Regardless of whether an operation is dispatched across a remote network socket to an OpenAI or OpenRouter endpoint, or routed through local inter-process communication to an on-device PyTorch, llama.cpp, or MLX worker, the input request schemas and output telemetry payloads adhere to a shared structural contract.

```python
class AIProvider:
    id: str

    def capabilities(self) -> ProviderCapabilities:
        ...

    def execute(self, request: NodeRequest) -> NodeResponse:
        ...

    def estimate_cost(self, request: NodeRequest) -> float:
        ...

    def benchmark(self, request: NodeRequest) -> TelemetryRecord:
        ...
```

Every execution returns an invariant telemetry packet containing four standardized metric dimensions:

* **Raw Output Payloads:** Normalized by modality into plain text strings, standardized floating-point vector arrays, binary audio buffers, or structured bounding boxes.
* **Elapsed Latency Tracking:** Recorded with millisecond precision from dispatch to final token or buffer delivery, isolating network transfer time from model processing time.
* **Financial Cost Accounting:** Capturing literal dollar expenditures derived from consumed prompt and completion tokens for cloud providers, or zero dollars for local hardware execution.
* **Hardware Telemetry Profiling:** Capturing local processor utilization, peak GPU VRAM allocation, and generation throughput measured in tokens per second or real-time factors for speech engines.

```json
{
  "output": "...",
  "usage": {
    "input_tokens": 120,
    "output_tokens": 80
  },
  "latency_ms": 820,
  "cost_usd": 0.002,
  "provider": "openrouter",
  "model": "anthropic/claude-3.5-sonnet",
  "hardware": {
    "tokens_per_second": 0,
    "vram_mb": 0
  }
}
```

---

## 3. Provider Capability Matrix and Dynamic Discovery

Heterogeneous AI providers exhibit vastly divergent functional capabilities. Proprietary model aggregators and specialized cloud platforms frequently support a subset of generative modalities, while local environments depend on installed execution binaries and physical GPU hardware. Noelle rejects static, hard-coded assumptions regarding provider features, implementing an active capability discovery engine.

The capability engine queries remote provider endpoints and scans the local environment upon startup to construct a live operational matrix across all registered backends.

| Provider / Target Engine | Large Language Models | Computer Vision & VLMs | Dense Embeddings | Speech-to-Text (STT) | Text-to-Speech (TTS) | Diffusion Image Generation |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Local Hugging Face / Transformers** | Fully Supported | Fully Supported | Fully Supported | Fully Supported | Fully Supported | Fully Supported |
| **Local llama.cpp Engine** | Fully Supported | Model Dependent | Fully Supported | Unsupported | Unsupported | Unsupported |
| **Local Ollama Runtime** | Fully Supported | Model Dependent | Fully Supported | Unsupported | Unsupported | Unsupported |
| **OpenAI Direct API** | Fully Supported | Fully Supported | Fully Supported | Fully Supported | Fully Supported | Fully Supported |
| **OpenRouter Aggregator** | Fully Supported | Model Dependent | Model Dependent | Unsupported | Unsupported | Model Dependent |
| **Specialized Cloud Runtimes** | Model Dependent | Model Dependent | Model Dependent | Fully Supported | Fully Supported | Fully Supported |

When a user constructs a workflow graph, the visual canvas actively validates node configurations against this matrix. If a user selects a provider that lacks native support for a chosen modality—such as attempting to assign an audio transcription task to an endpoint that only provides text completion—the interface flags the incompatibility, prevents execution dispatch, and suggests compatible local or remote alternatives.

---

## 4. Multi-Modal Workflow Node Specifications

The Hybrid Workflow Lab provides dedicated functional nodes across six core machine learning disciplines. Each node encapsulates model parameterization, execution routing, and telemetry capture.

### 4.1 Large Language Model Node
The language model node handles text generation, conversational chat, structured JSON emission, and tool calling. Users specify system prompts, generation sampling hyper-parameters including temperature and top-p, context truncation boundaries, and token limits. When connected to cloud providers, the node supports streaming Server-Sent Events alongside standard batch completions. When assigned to local execution, the node interfaces with Noelle's out-of-process PyTorch or llama.cpp workers, enforcing local context clamping based on detected VRAM headroom.

### 4.2 Dense Text Embedding Node
The embedding node accepts single strings or batched document arrays, converting text into normalized dense vectors. The node supports dimensionality matching, pooling configuration (mean, CLS, or last-token pooling), and cosine distance scoring. Practitioners utilize this node to evaluate on-device embedding architectures against hosted cloud embedding services across identical corpus partitions.

### 4.3 Speech-to-Text Transcription Node
The speech recognition node processes raw waveform inputs from local audio files or direct microphone capture. The node exposes language selection, timestamp generation toggles, and temperature fallbacks. Local execution routes requests to native Whisper pipelines or specialized speech checkpoints, while cloud execution targets hosted transcription endpoints.

### 4.4 Text-to-Speech Synthesis Node
The speech synthesis node transforms text into audible speech buffers. Controls include speaker voice selection, pitch, speaking rate, and sample rate configuration (16 kHz, 24 kHz, or 48 kHz). The node outputs audio waveforms alongside objective metrics including generation duration, real-time factor, and audio length.

### 4.5 Computer Vision and Multimodal Node
The vision node processes visual inputs paired with textual queries, supporting document optical character recognition (OCR), visual question answering, and bounding box object localization. Users can pass complex visual artifacts, such as digitized invoices or architectural diagrams, to evaluate extraction precision across local vision-language models and frontier cloud multimodal APIs.

### 4.6 Retrieval-Augmented Generation (RAG) Subgraph Node
The RAG node encapsulates a complete retrieval and context-injection pipeline. It coordinates document chunking, vector database querying using an embedded sqlite-vec index, context pruning to a specified token budget, and prompt assembly prior to passing augmented prompts to a downstream language model node. This composite node enables modular substitution of the embedding engine, the vector store, or the generation model independently.

---

## 5. Empirical Benchmarking and Evaluation Laboratories

The distinguishing strength of the Hybrid Lab is its capacity to run controlled empirical experiments comparing cloud endpoints and local models under identical inputs. Rather than relying on subjective manual testing, Noelle provides specialized evaluation suites tailored to specific modalities.

### 5.1 Local vs. Cloud Embedding Evaluation Lab
Evaluating embedding performance is critical for determining whether an enterprise requires an expensive hosted embedding API or can achieve sufficient semantic retrieval using a lightweight local model. The Embedding Lab ingests a standardized validation corpus—such as 10,000 domain-specific document passages paired with ground-truth search queries—and computes rigorous retrieval metrics across competing backends.

| Embedding Engine | Deployment Model | Recall@10 Metric | Average Latency | VRAM / RAM Footprint | Direct API Cost per 10k Docs |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **OpenAI text-embedding-3-small** | Cloud API | 92.4% | 320 ms | 0 MB (Remote) | Metered Tier Cost |
| **BAAI/bge-m3 (FP16)** | Local CUDA Worker | 91.2% | 81 ms | ~2.2 GB VRAM | $0.00 |
| **intfloat/e5-base-v2 (INT8)** | Local CUDA Worker | 88.6% | 64 ms | ~1.1 GB VRAM | $0.00 |
| **jinaai/jina-embeddings-v3** | Local CUDA Worker | 91.8% | 93 ms | ~2.4 GB VRAM | $0.00 |

The evaluation suite charts precision-recall curves, measures semantic similarity distributions, and demonstrates that local embedding models frequently achieve comparable recall while slashing latency by over 70% due to the elimination of network transmission overhead.

### 5.2 Speech Recognition (STT) Robustness Lab
The Speech Recognition Lab evaluates transcription accuracy by passing standardized audio datasets through local Whisper variants and cloud transcription APIs. To test operational resilience, the lab generates adversarial evaluation cases featuring varied acoustic challenges:
* **High-Density Numerical Sequences:** Evaluates precision across financial statements, currency denominations, timestamps, and serial identifiers.
* **Domain-Specific Technical Jargon:** Evaluates accuracy when transcribing legal terminology, medical phrasing, and software engineering syntax.
* **Multi-Lingual Code-Switching:** Evaluates how accurately engines transcribe natural dialogues that blend regional languages (such as Indonesian) with technical English terms.

The lab programmatically calculates Word Error Rate (WER) and Character Error Rate (CER), pairing these figures with processing latency and Real-Time Factor (RTF) calculations.

### 5.3 Speech Synthesis (TTS) Objective and A/B Evaluation Lab
The Text-to-Speech Lab provides a side-by-side listening station paired with deterministic acoustic metrics. When identical text inputs are processed by competing local engines (such as XTTS or SpeechT5) and cloud voice endpoints, Noelle measures:
* Generation latency and Real-Time Factor (RTF = total processing time / total audio duration).
* Spectral bandwidth and output sample rate.
* Pronunciation confidence and word-level temporal alignment.

The interface incorporates a blind A/B playback console where human evaluators can audition audio clips without knowing the generating provider, recording qualitative scores that are saved directly into the experiment record.

### 5.4 Multimodal Vision Extraction Lab
The Vision Lab focuses on practical document understanding, such as extracting structured line items from scanned invoice images. The evaluation harness inputs identical high-resolution documents to cloud vision models and local vision-language models (such as Qwen-VL or SmolVLM). Extracted key-value pairs are validated against expected ground-truth JSON schemas, measuring extraction accuracy, numeric fidelity, and processing speed.

---

## 6. Cost Accounting and the Cloud vs. Local Decision Engine

A major operational hurdle in production AI architecture is predicting deployment expenses when scaling from prototype to production. Noelle incorporates a real-time Cost Accounting Engine paired with a predictive Cloud vs. Local Decision Engine.

### 6.1 Granular Cost Accounting
For every execution routed through cloud providers, Noelle parses billing headers and usage metadata, calculating exact expenditures based on published pricing:

$$\text{Cost}_{\text{cloud}} = (\text{Tokens}_{\text{input}} \times \text{Rate}_{\text{input}}) + (\text{Tokens}_{\text{output}} \times \text{Rate}_{\text{output}})$$

For local execution, direct API costs are tracked as exactly zero dollars. However, users can define local infrastructure parameters, including average hardware power draw in watts, local kilowatt-hour electricity costs, and hardware purchase amortization schedules to calculate the true total cost of ownership (TCO).

### 6.2 The Automated Decision Engine
The Decision Engine synthesizes empirical benchmark results with user-defined deployment parameters to provide actionable infrastructure recommendations.

| Metric Dimension | Cloud API Deployment (OpenAI / OpenRouter) | Dedicated Local Workstation (NVIDIA RTX 4090) | Operational Differential & Impact |
| :--- | :--- | :--- | :--- |
| **Direct API Costs** | $2,450.00 / month | $0.00 / month | 100% elimination of variable token billing |
| **Local Electricity Overhead** | $0.00 / month | $48.50 / month (at $0.15 / kWh, 450W load) | Minor incremental utility overhead |
| **Hardware Capital Amortization** | $0.00 / month | $83.33 / month ($2,000 amortized over 24 mo) | Fixed, predictable monthly asset cost |
| **Total Monthly Net Cost** | $2,450.00 / month | $131.83 / month | **$2,318.17 monthly savings (94.6% reduction)** |
| **Average End-to-End Latency** | 410 ms (includes network round-trip) | 145 ms (direct PCIe memory bus transfer) | Local processing executes 2.8x faster |
| **Data Sovereignty & Egress** | Transmitted to third-party data centers | Zero Egress (retained entirely on-device) | Verifiable regulatory compliance |
| **Economic Break-Even Point** | Reached at ~8,200 requests / day | Immediate cost parity achieved at day 9 | Permanent cost ceiling established |

---

## 7. Privacy Modes and Data Transmission Guardrails

Because Noelle handles sensitive enterprise documentation, proprietary data, and private API keys, the Hybrid Workflow Lab enforces strict privacy governance through three explicit operational modes.

### 7.1 Local-Only Mode (Zero Egress Guarantee)
In Local-Only Mode, the desktop application completely severs outbound network connections for model processing. All cloud provider nodes are hard-disabled within the canvas. The local inference worker runs in an isolated operating system sandbox with `HF_HUB_OFFLINE=1`. Under no circumstances is user prompt text, document content, vector data, or audio transmitted outside the physical machine.

### 7.2 Hybrid Mode (Controlled Egress with Explicit Consent)
Hybrid Mode permits the execution of graphs containing both local runtimes and remote cloud APIs, subject to strict boundary guardrails:
* **Visual Node Tagging:** Cloud nodes are prominently rendered with distinct visual borders, cloud badges, and destination endpoint indicators.
* **Pre-Dispatch Payload Inspection:** Prior to dispatching a pipeline that routes data to an external API, the interface displays an inspection modal detailing the exact payload content, destination URL, and transmitting provider.
* **Granular Transmission Authorization:** Users can whitelist specific non-sensitive pipeline stages for cloud execution while permanently constraining sensitive document indexing to local nodes.

### 7.3 Cloud-Only Mode
In Cloud-Only Mode, pipelines route entirely through hosted third-party endpoints, serving developers prototyping application logic on low-powered client devices before transferring workflows to local workstations.

---

## 8. Automated Test Generation and Adversarial Validation

Evaluating artificial intelligence pipelines using trivial or static test inputs leads to over-optimistic performance assessments that fail under production conditions. Noelle embeds an automated test synthesis engine that programmatically creates domain-specific validation suites.

### 8.1 Automated Test Dataset Synthesis
Users define testing criteria by specifying domain targets, input modalities, and complexity distributions:
* **Linguistic Stress Tests:** Synthesizes passages containing complex phonetic homophones, grammatical edge cases, and multi-lingual idioms to evaluate speech and translation fidelity.
* **Numeric and Entity Precision:** Formulates prompts requiring exact arithmetic reasoning, tabular extraction, and strict date formatting to test factual extraction consistency between local and cloud language models.
* **Adversarial Context Ingestion:** Constructs RAG evaluation passages containing distracting, conflicting, or semantically adjacent facts to test whether vector search and context-pruning algorithms preserve grounding without hallucinations.

### 8.2 TTS Difficult Sentence Generator (Indonesian & Technical Focus)
* **Numbers & Currency:** `"Pada tahun 2026, pendapatan meningkat sebesar Rp17.845.230,50 atau sekitar 13,72 persen."`
* **Technical Terms:** `"PyTorch menggunakan CUDA 12.4 dengan tensor bertipe bfloat16 pada GPU NVIDIA."`
* **Acronyms:** `"API REST menggunakan OAuth 2.0 dan JWT untuk mengakses endpoint HTTPS."`
* **Proper Nouns:** `"Dr. Raden Mas Soerjono Kartohadiprodjo menghadiri konferensi di Yogyakarta."`
* **Code-Switching:** `"Model tersebut menggunakan Retrieval-Augmented Generation dengan hybrid search dan reranking."`
* **Phonetic Challenges:** `"Konfigurasi tersebut memprioritaskan sinkronisasi, standardisasi, dan interoperabilitas."`

### 8.3 Workflow Replay and Batch Execution
Every workflow graph and dataset combination is saved as an executable, reproducible YAML manifest:

```yaml
workflow:
  name: voice-assistant-test
  privacy_mode: hybrid

providers:
  openrouter:
    model: anthropic/claude-3.5-sonnet
  local:
    stt_model: openai/whisper-large-v3
    tts_model: suno/bark-small

nodes:
  - id: input_audio
    type: input_audio
  - id: transcribe
    type: stt
    provider: local
    model: openai/whisper-large-v3
  - id: generate
    type: llm
    provider: openrouter
    model: anthropic/claude-3.5-sonnet
  - id: speak
    type: tts
    provider: local
    model: suno/bark-small

dataset:
  path: ./datasets/voice-test.jsonl
```

Executed via GUI or headless CLI:
```bash
noelle eval --workflow ./workflows/voice-assistant-test.yaml --dataset ./datasets/voice-test.jsonl
```

---

## 9. Phased Implementation Roadmap and Acceptance Criteria

### Phase 1: Core Provider Abstraction and Dual-Node Workflow Canvas
* Implement `ProviderInterface` base class and normalize request/response telemetry schemas.
* Integrate direct OpenAI and OpenRouter client adapters alongside the local Noelle worker client.
* Secure API key management integrated with the native operating system credential vault (Keychain, Windows Credential Manager, Secret Service).
* Build visual dual-node canvas in PySide6, supporting sequential chaining of language model and embedding nodes across cloud and local targets.
* **Acceptance Gate:** User can construct a pipeline that routes an input prompt through a local embedding model, queries local sqlite-vec storage, passes context to a cloud language model, and displays latency and cost metrics alongside generated text.

### Phase 2: Full Multimodal Node Suite and Empirical Embedding Lab
* Implement dedicated nodes for Speech-to-Text (Whisper local vs. cloud), Text-to-Speech (local acoustic synthesis vs. cloud voice APIs), and Computer Vision.
* Release the dedicated Embedding Evaluation Lab, supporting corpus ingestion, automated Recall@K scoring, and latency-cost comparison tables.
* Establish visual canvas validation against the dynamic Provider Capability Matrix.
* **Acceptance Gate:** System successfully benchmarks a local BGE-M3 embedding model against an external cloud API across 5,000 document passages, exporting a reproducible comparison report containing recall curves, latency distributions, and total cost metrics.

### Phase 3: Cost Accounting Engine and Cloud vs. Local Decision Advisor
* Build automated token and billing calculation service for all registered cloud providers.
* Integrate local hardware power profiling and electricity cost estimation into the telemetry pipeline.
* Implement predictive Decision Engine calculating break-even volume curves and recommending optimal local, cloud, or hybrid deployment topologies.
* Introduce Privacy Mode enforcement, including visual node tagging and pre-dispatch confirmation modals for hybrid pipelines.
* **Acceptance Gate:** Entering a hypothetical workload of 250,000 monthly queries yields an automated feasibility assessment displaying break-even timelines, estimated operational costs, and latency differentials between local workstation execution and cloud hosting.

### Phase 4: Automated Test Generation, Batch Workflow Replay, and CI Export
* Build domain-specific automated test case synthesis engine (TTS/STT adversarial Indonesian & technical generator).
* Implement batch execution coordinator with rate-limiting throttles for cloud endpoints and sequential VRAM safety queues for local workers.
* Expose workflow execution and benchmark replay through the Noelle command-line interface (`noelle eval --workflow <path> --dataset <path>`).
* **Acceptance Gate:** Executing a batch workflow replay across 1,000 multimodal test cases completes without GUI thread blocking, capturing comprehensive experiment artifacts in SQLite and outputting standardized JSON and Markdown evaluation summaries.
