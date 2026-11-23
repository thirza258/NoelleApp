---
title: Terms of Service
nav: Terms of Service
description: Terms of service, acceptable use guidelines, open source license terms, and disclaimers for Noelle.
section: Reference
order: 31
slug: terms
---

**Effective Date:** November 1, 2026 &bull; **Version:** 1.1  
**Project:** Noelle ([noelle.nevatal.id](https://noelle.nevatal.id)) &bull; **Ecosystem:** [nevatal.id](https://nevatal.id)

These Terms of Service ("Terms") govern your use of the Noelle desktop application, local runtime services, developer APIs, and documentation website.

---

## 1. Agreement to Terms & Open Source Licensing

Noelle is free, open-source software distributed under the **[MIT License](https://opensource.org/licenses/MIT)**. By downloading, compiling, installing, or utilizing Noelle, you agree to these Terms. If you do not agree, do not use the application or website.

The source code is publicly accessible and inspectable on [GitHub](source:). Your right to inspect, fork, modify, and redistribute the application is guaranteed under the MIT License.

---

## 2. Local Architecture & Operational Scope

Noelle is provided as an on-device model laboratory and local inference gateway:

- **Local Execution:** Model execution, quantization, evaluation benchmarking, and vector embeddings operate on your own computing hardware.
- **Non-Commercial Public Utility:** The project is published on a free, non-profit basis for developers, researchers, and students.
- **Zero Account Obligation:** You are not required to create an account, register personal data, or provide telemetry to operate Noelle.

---

## 3. Acceptable Use Guidelines

You agree to utilize Noelle in compliance with all applicable local, national, and international laws. Specifically, you agree **NOT** to use Noelle or its local developer APIs to:

1. **Malicious Payloads & Cyber Exploits:** Generate, optimize, or distribute malware, ransomware, exploit payloads, or malicious scripts.
2. **Deceptive Content & Harassment:** Conduct phishing schemes, non-consensual deepfakes, automated defamatory campaigns, or harassment.
3. **Infrastructure Disruption:** Initiate denial-of-service (DoS) attacks, unauthorized network probing, or abusive automated scraping against the Noelle documentation website or third-party package mirrors.
4. **License Infringement:** Disregard specific restrictions set forth by model publishers (e.g., Llama Community License, DeepSeek Terms, or Mistral AI licenses).

---

## 4. Third-Party Models, Code & Repositories

Noelle interfaces with open ecosystem registries to facilitate model discovery and download:

- **Hugging Face Hub:** When browsing or downloading model weights from `huggingface.co`, you are responsible for adhering to the [Hugging Face Terms of Service](https://huggingface.co/terms-of-service) and the individual licenses specified by model repositories.
- **Custom Model Code Execution:** Repositories distributing executable Python scripts (`trust_remote_code=True`) pose inherent security risks. Noelle quarantines custom code until you explicitly authorize execution. You assume full responsibility for running community-authored scripts. See [Models & builds](docs:models#repositories-that-run-their-own-code).
- **llama.cpp Binaries:** Optional native runner binaries are downloaded from official GitHub releases only upon your direct confirmation.

---

## 5. Intellectual Property & User Output Ownership

- **Prompts & Datasets:** You retain complete ownership and intellectual property rights over all prompts, documents, datasets, and custom evaluation benchmarks you create or import into Noelle.
- **Model Generations:** Subject to the underlying licenses of the AI models you run, all generated text, code, embeddings, and fine-tuned LoRA weights produced locally on your hardware belong to you.
- **No Claims by Noelle:** Noelle claims zero ownership, royalties, or licensing rights over content generated through your local deployments.

---

## 6. Disclaimer of Warranties

NOELLE AND ITS ACCOMPANYING DOCUMENTATION ARE PROVIDED **"AS IS"** AND **"AS AVAILABLE"**, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, TITLE, AND NON-INFRINGEMENT.

AI models are probabilistic software systems. The operators and maintainers of Noelle do not warrant that model completions, benchmark scores, or diagnostic outputs will be error-free, uninterrupted, or suitable for critical medical, legal, or financial decision-making.

---

## 7. Limitation of Liability

IN NO EVENT SHALL THE AUTHORS, COPYRIGHT HOLDERS, OR HOST OPERATORS BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT, OR OTHERWISE, ARISING FROM, OUT OF, OR IN CONNECTION WITH THE SOFTWARE, THE USE OF LOCAL OR REMOTE MODELS, OR THE INABILITY TO USE THE SERVICE.

---

## 8. Modifications to Terms

We reserve the right to revise or update these Terms to reflect architectural enhancements or legal standards. Updated versions will be published to [noelle.nevatal.id/docs/terms/](https://noelle.nevatal.id/docs/terms/) with a revised effective date.

---

## 9. Privacy Policy

Our data protection principles, process isolation boundaries, and credential security measures are detailed in our [Privacy Policy](docs:privacy).

---

## 10. Contact & Legal Notices

For legal inquiries, terms clarifications, or compliance reports, please contact:  
📧 **[nevatalapp@gmail.com](mailto:nevatalapp@gmail.com)**
