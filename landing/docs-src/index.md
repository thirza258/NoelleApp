---
title: Noelle documentation
nav: Overview
description: Learn how to install Noelle, find and run open AI models locally, evaluate them on your own tasks and use them from your apps.
section: Get started
order: 1
slug: ""
---

Noelle is a free desktop app for macOS, Windows and Linux that runs open AI models from the Hugging Face Hub
**on your own computer**. It helps you pick models that fit your hardware, run them, measure how good they are at
your tasks, and use them from other apps — without sending your data anywhere.

<div class="doc-cards">
  <a href="docs:install"><b>Install Noelle</b><span>Installers for macOS, Windows and Linux, and the setup assistant.</span></a>
  <a href="docs:quickstart"><b>Quick start</b><span>From download to your first chat in a few minutes.</span></a>
  <a href="docs:mission"><b>Mission mode</b><span>Describe a goal and let Noelle propose and build the stack.</span></a>
  <a href="docs:models"><b>Models &amp; builds</b><span>Compatibility scores, run options, profiles and 4-bit builds.</span></a>
  <a href="docs:evaluation"><b>Evaluation Lab</b><span>Compare models on datasets and keep a history of every run.</span></a>
  <a href="docs:knowledge"><b>Knowledge</b><span>Chat with your documents and measure retrieval quality.</span></a>
  <a href="docs:api"><b>Local API</b><span>OpenAI-compatible endpoints, aliases and automatic routing.</span></a>
  <a href="docs:mcp"><b>MCP server</b><span>Let Claude Code, Cursor and other agents use your local models.</span></a>
</div>

## How Noelle is organised

| Page | What you do there |
|---|---|
| **Discover** | Browse the Hugging Face catalog. Every model shows a 0–100 score for your machine. |
| **Mission** | Describe what you want to build; Noelle proposes models and sets them up step by step. |
| **Workspace** | One model at a time: analysis, download, runtime settings and a playground for its type. |
| **Knowledge** | Knowledge bases built from your documents, search and retrieval evaluation. |
| **Evaluate** | Run datasets against several models; history and a performance matrix. |
| **Compare** | One prompt, up to four models side by side. |
| **Storage** | Disk usage, the Storage Advisor and clean-up tools. |
| **API** | The local OpenAI-compatible server, aliases and ready-to-copy snippets. |

## What runs where

Noelle has two parts. The **app** you see handles browsing, downloads and settings. The **model runtime** is a
separate process that loads models and generates answers. The runtime works offline and never receives your
Hugging Face token, and if a model runs out of memory, only the runtime restarts — the app keeps your work.

> New to local AI? Start with the [quick start](docs:quickstart). It walks you through your first model in a few
> minutes.
