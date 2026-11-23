---
title: Privacy & security
nav: Privacy & security
description: What Noelle sends over the network, where your data and token are stored, and how the local API and model runtime are isolated.
section: Reference
order: 30
slug: privacy
---

Noelle is built so that **your prompts, documents and answers never leave your computer**.

## What goes over the network

| When | Where to | What |
|---|---|---|
| Browsing Discover, opening a model | huggingface.co | Searches and model metadata |
| Downloading a model | huggingface.co | The model files (verified by checksum) |
| Gated models | huggingface.co | Your Hugging Face token, only if you added one |
| Optional components | github.com | The official llama.cpp release, only after you confirm |
| Web search in chat (optional) | Your own SearXNG server | The search queries |

Noelle has **no telemetry** and no account, and it turns off telemetry in the libraries it uses.

## The model runtime

Models run in a separate process. It is started with Hugging Face's offline mode switched on and **never receives
your Hugging Face token**. Commands between the app and the runtime use a random per-session key over the local
machine only.

## Your token

The Hugging Face token is stored in the operating system's credential store — Keychain on macOS, Credential Manager
on Windows, Secret Service on Linux — never in a settings file or log.

## The local API

- Listens on `127.0.0.1` only.
- Answers only requests addressed to `127.0.0.1`, `localhost` or `[::1]`, so a website cannot reach it through DNS
  rebinding.
- JSON endpoints require `Content-Type: application/json`, so web pages cannot send it simple form posts.
- An optional API key protects it from other programs on the same computer.

## Code from model repositories

Repositories that ship their own Python code are blocked until you allow them — once, per repository or per
publisher. See [Models & builds](docs:models#repositories-that-run-their-own-code).

## Downloads you didn't ask for

There are none. Optional components such as the llama.cpp tools show their version, size, source and destination
and wait for your confirmation; their checksums are verified before installation.
