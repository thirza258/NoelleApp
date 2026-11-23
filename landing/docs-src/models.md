---
title: Models & builds
nav: Models & builds
description: Understand the compatibility score, choose a run option and runtime profile, build 4-bit versions and decide which repositories may run their own code.
section: Using Noelle
order: 11
slug: models
---

## The compatibility score

Every model gets a score from **0 to 100** for your computer. The **Analysis** tab in the Workspace shows how it is
made up:

| Part | Points | What it checks |
|---|---|---|
| **Memory fit** | 40 | Whether the recommended build fits in usable memory, with room to spare |
| **Runtime** | 20 | Whether an engine for this model and file format runs on your device |
| **Precision** | 15 | Whether the recommended precision is ready, needs a one-time build or an optional component |
| **Context** | 10 | How long a conversation or document the model can handle |
| **Trust & access** | 15 | Whether it runs its own Python code, is gated, or has no licence |

A model that cannot run on your device at all is capped at a low score, whatever its other qualities.

## How should I run this?

The Analysis tab lists every way to run the model:

| Build | Engine | Where |
|---|---|---|
| FP16 / BF16 | Transformers | NVIDIA, Apple Silicon (FP32 on CPU) |
| 8-bit, 4-bit NF4 | Transformers + bitsandbytes | NVIDIA only |
| 4-bit AWQ, 4-bit GPTQ | Transformers | NVIDIA only (built locally) |
| 4-bit MLX | MLX | Apple Silicon (built locally) |
| GGUF Q4_K_M and other GGUF files | llama.cpp | Everywhere |

For each one you see the memory it needs, whether it fits, the expected quality (reference, near-lossless, small
loss, noticeable loss) and the speed. Speeds you have **measured** on this machine — in Compare or the Evaluation
Lab — replace the estimates. Each row also says whether the build is **ready**, needs a **one-time build**, needs an
**optional component**, or **can't run on this device**, with the reason.

## Runtime profiles

The **Profile** menu chooses the build for you:

| Profile | Chooses |
|---|---|
| **Balanced** | Full precision when it fits comfortably, otherwise the best 4-bit build for your device |
| **Quality** | The highest-precision build that fits |
| **Fast** | The smallest weights on the fastest engine for your device |
| **Low memory** | The smallest memory footprint |
| **Max context** | Small weights so the longest possible context fits |

## Building 4-bit versions

Open the **Optimize** tab to build a local 4-bit copy: **4-bit MLX** on Apple Silicon, **GGUF Q4_K_M** with the
llama.cpp tools, and **AWQ**, **GPTQ** or **NF4** on NVIDIA GPUs. The new build appears in the **Build** menu and
in every other page.

GGUF needs the official llama.cpp tools. Noelle offers to download them the first time you need them, shows the
version, size and source, asks first, and verifies the published checksum. You can also manage them in
**Settings › Components**.

## LoRA adapters

For text models, the Optimize tab can attach LoRA adapters to a loaded model and switch between them without
reloading the base weights.

## Repositories that run their own code

Some repositories ship Python files that run when the model loads. Noelle lists those files and blocks them until
you decide:

- **Allow once** — for this session only.
- **Trust repository** — remember this repository.
- **Trust publisher** — remember every repository from this organisation.

Your choice applies everywhere the model runs: the Workspace, Compare, the Evaluation Lab, Mission mode, the
command line and the API. **Review files** opens the code on huggingface.co; **Revoke** takes the trust back.
