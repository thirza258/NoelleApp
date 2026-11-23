---
title: Privacy & security
nav: Privacy & security
description: How Noelle guarantees 100% local processing, zero telemetry, secure credential storage, and network transparency.
section: Reference
order: 30
slug: privacy
---

**Effective Date:** November 1, 2026 &bull; **Version:** 1.1  
**Project:** Noelle ([noelle.nevatal.id](https://noelle.nevatal.id)) &bull; **Ecosystem:** [nevatal.id](https://nevatal.id)

Noelle is built on the principle of **local data sovereignty**: your prompts, documents, embeddings, and generated answers **never leave your computer**.

---

## 1. Local Data Boundary & Sovereignty Guarantee

Noelle is a 100% local desktop model laboratory and inference gateway. Unlike cloud AI platforms and software-as-a-service providers that transmit your sensitive data over the internet:

- **Local Inference:** All model generation, summarization, and chat completions execute directly on your local hardware (CPU/GPU) using isolated native runtimes (such as llama.cpp and local backends).
- **Local Knowledge & Documents:** Any files, PDFs, notes, or codebases you index for RAG (Retrieval-Augmented Generation) are chunked and embedded into local vector indexes stored on your own disk.
- **Zero Prompt Leakage:** No prompt, document fragment, chat transcript, or evaluation metric is ever sent to external cloud AI providers (such as OpenAI, Anthropic, or OpenRouter) unless you deliberately configure external remote providers.
- **Zero Telemetry & Tracking:** Noelle contains **no telemetry**, no analytics trackers, no advertising pixels, and requires no account. Furthermore, Noelle explicitly disables background telemetry in third-party Python libraries (such as Hugging Face Hub telemetry).

---

## 2. What Goes Over the Network

Noelle communicates with external networks strictly for operations that you explicitly initiate:

| Operation | Destination | Data Sent | Security & Verification |
|---|---|---|---|
| **Browsing Discover** | `huggingface.co` | Model search keywords and metadata requests | Encrypted HTTPS; anonymous requests |
| **Downloading Models** | `huggingface.co` | Standard GET requests for model weight files | HTTPS; verified against published SHA256 checksums |
| **Gated / Restricted Models** | `huggingface.co` | Your personal Hugging Face user access token | Transmitted securely over HTTPS only when authorized |
| **Optional Components** | `github.com` | GET requests for official llama.cpp releases | Explicit user prompt; checksums validated before execution |
| **Web Search in Chat (Optional)** | User-configured SearXNG | Search query strings | Directed only to your private self-hosted SearXNG endpoint |

There are **zero automatic or unsolicited background downloads**. Optional tools and runtime updates always display their file size, origin repository, destination path, and SHA256 checksum, requiring your manual confirmation.

---

## 3. Runtime & Process Isolation

Models run in a dedicated, isolated subprocess separate from the primary user interface:

- **Offline Mode Enforcement:** The runtime process starts with Hugging Face offline mode enabled (`HF_HUB_OFFLINE=1`), ensuring it cannot make unauthorized outbound connections during inference.
- **Token Isolation:** The runtime process **never receives your Hugging Face API token** or system credentials.
- **Local Inter-Process Communication:** Communication between the desktop application and the worker service is confined to localhost using an ephemeral, cryptographically random per-session token.

---

## 4. Credential & Secret Management

- **Operating System Credential Vaults:** Your Hugging Face user access tokens and optional API keys are stored exclusively in your operating system's native secure credential storage:
  - **macOS:** Keychain Services
  - **Windows:** Windows Credential Manager
  - **Linux:** Secret Service API (Freedesktop Secret Service / Keyring)
- **No Plaintext Leaks:** Credentials are never written to plain-text configuration files, debug logs, or crash dumps.

---

## 5. Local API Security & DNS Rebinding Protection

Noelle includes an OpenAI-compatible local HTTP server for developer integrations (CLI, MCP, and external coding agents):

- **Localhost Binding:** Binds strictly to loopback interfaces (`127.0.0.1` and `::1`).
- **Host Header Validation:** Rejects any HTTP request whose `Host` header does not match `127.0.0.1`, `localhost`, or `[::1]`, neutralizing DNS rebinding attacks from malicious websites.
- **Content-Type Enforcement:** Endpoints require strict `Content-Type: application/json` headers to prevent cross-origin form submissions.
- **Optional API Key Authentication:** An optional local bearer token can be configured to restrict API access from other unprivileged programs on the local system.

---

## 6. Model Code Execution Safeguards

Certain model repositories on Hugging Face distribute custom Python execution scripts (`trust_remote_code=True`). Noelle blocks automated execution of arbitrary repository code by default:

- Remote code execution is quarantined until you explicitly review and grant permission.
- Permissions can be granted once, remembered per-repository, or allowed per-publisher. See [Models & builds](docs:models#repositories-that-run-their-own-code).

---

## 7. Website, Cookies & Server Logs (Landing Page & Docs)

When visiting the Noelle website ([noelle.nevatal.id](https://noelle.nevatal.id)):

- **No Cookies or Trackers:** We do not use advertising cookies, third-party analytics (like Google Analytics), or cross-site tracking beacons.
- **Local Storage:** The website uses browser `localStorage` solely to persist your theme preference (`light` or `dark`).
- **Server Access Logs:** Our self-hosted Nginx web server records standard transient connection logs (IP address, user agent, request timestamp) strictly for DDoS mitigation, caching, and performance diagnostics. These logs are never merged with personal identifiers or shared with third parties.

---

## 8. Data Retention, Portability & User Rights

Because all data resides locally on your hardware, you retain 100% control over your assets:

- **Model Weights & Cache:** You can purge downloaded models or individual quantizations directly through the app's Storage tab or by deleting `~/.cache/noelle/`.
- **Chat Histories & Evaluations:** Benchmark runs and conversation sessions can be cleared or exported in standard formats at any time.
- **Credential Removal:** You can revoke or wipe saved Hugging Face tokens instantly via the Setup Assistant or your OS credential manager.

---

## 9. Contact & Inquiries

Noelle is an open-source project published under the MIT License within the **[nevatal.id](https://nevatal.id)** ecosystem. See also our [Terms of Service](docs:terms).

For privacy inquiries, security disclosures, or questions regarding this deployment, please contact us directly at:  
📧 **[nevatalapp@gmail.com](mailto:nevatalapp@gmail.com)**
