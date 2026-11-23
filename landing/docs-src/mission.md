---
title: Mission mode
nav: Mission mode
description: Describe what you want to build and let Noelle propose a local AI stack that fits your hardware, then set it up step by step.
section: Using Noelle
order: 10
slug: mission
---

Mission mode turns a goal written in plain language into a working local setup. Open **Mission**, describe what you
want to build and click **Plan mission**.

## What to write

Describe the task, the languages and anything you know about the hardware. For example:

- *A local Indonesian legal assistant for my PDF contracts*
- *A fast coding assistant for Python*
- *Transcribe my meeting recordings*
- *I need Indonesian document QA. I have 16 GB RAM and an RTX 3060.*

Noelle recognises:

| From your goal | Examples |
|---|---|
| **Task** | chat, coding, questions over documents (RAG), translation, image understanding, speech to text, text to speech, image generation, semantic search |
| **Languages** | Indonesian, English, Chinese, Japanese, Javanese, Sundanese, Malay and more — or “multilingual” |
| **Priorities** | “fast”, “best quality”, “low memory”, “long documents” |
| **Licence needs** | “commercial”, “for my company” — restrictive licences are ranked lower |
| **Hardware** | “RTX 3060”, “8 GB VRAM”, “M2 with 24 GB” — otherwise Noelle uses this computer |

## The plan

Noelle searches the Hugging Face Hub, scores every candidate for the target hardware and proposes a stack:

- a **language model** with the build to use (for example 4-bit NF4 on an NVIDIA card or 4-bit MLX on a Mac),
- an **embedding model** and a local **vector store** when you work with documents,
- the right model for speech, images or vision tasks,
- the **estimated memory** for the whole stack, and **alternatives** you can switch to.

Each choice lists its reasons: size and score on your hardware, language support, whether it is trained for code,
whether it is already downloaded.

## Running the plan

The plan is a list of steps you can run one by one or all together:

1. **Download** the models.
2. **Build** a 4-bit version when the recommended build has to be made locally.
3. **Index** your documents into a knowledge base (document tasks only).
4. **Test** the language model on a matching built-in dataset; the result is saved in the
   [Evaluation Lab](docs:evaluation).
5. **Serve** it on the [local API](docs:api) under an alias such as `noelle-chat` or `noelle-code`.

Every step that downloads something, converts a model or starts the server **asks you first**. If you decline a
step, the plan stops there. Click **Export plan** to save the plan as Markdown.

> Mission mode is a transparent, rule-based planner — no hidden AI decides for you. Every recommendation comes from
> the same compatibility scores you see on Discover.
