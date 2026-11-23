---
title: Quick start
nav: Quick start
description: Find a model that fits your computer, download it and chat with it — in a few minutes.
section: Get started
order: 3
slug: quickstart
---

This walkthrough takes you from a fresh install to a working local chat model.

## 1. Find a model that fits

Open **Discover** and choose **Text & Language**. Every card has a badge with a **score from 0 to 100** for your
computer:

- **Green** — runs natively at full precision.
- **Yellow** — needs a 4-bit build to fit in memory.
- **Red** — too large, or only a slow CPU fallback.

Sort by **Most downloads** to see well-known models first. Good first picks on a 16 GB machine are models with
1–4 billion parameters, such as `Qwen/Qwen2.5-1.5B-Instruct` or `HuggingFaceTB/SmolLM2-1.7B-Instruct`.

## 2. Check the analysis

Click a card to open it in the **Workspace**. The sidebar shows the score; the **Analysis** tab explains it and
lists every way to run the model (full precision, 8-bit, 4-bit MLX, GGUF, …) with the memory each needs. See
[Models & builds](docs:models) for details.

## 3. Download

Click **Download** in the sidebar. Downloads resume if interrupted and every file is checked against its
checksum. You can follow progress in the top bar or on the **Storage** page.

## 4. Load and chat

When the download finishes, click **Load model**, type a message in the **Text playground** and press **Enter**.
Each answer shows the time to first token and the speed in tokens per second.

> Tip: the **Profile** menu (Balanced, Quality, Fast, Low memory, Max context) picks the best build for you. If the
> recommended build needs to be made first, Noelle tells you and the **Optimize** tab builds it in one click.

## 5. Next steps

- Let Noelle choose the models for a whole project with [Mission mode](docs:mission).
- Find out which model is best at your tasks in the [Evaluation Lab](docs:evaluation).
- Chat with your own documents using [Knowledge](docs:knowledge).
- Use the model from your code with the [local API](docs:api).
