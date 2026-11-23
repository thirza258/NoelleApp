---
title: Install Noelle
nav: Install
description: Install Noelle on macOS, Windows or Linux, run the setup assistant and add a Hugging Face token.
section: Get started
order: 2
slug: install
---

Noelle ships as a regular installer for each platform. Get the latest version from the
[download page](download:) — it picks the right file for your system — or browse
[all installers and older versions](files:).

## System requirements

| | Minimum | Recommended |
|---|---|---|
| **macOS** | macOS 13 on Apple Silicon (M1 or newer) | 16 GB unified memory |
| **Windows** | Windows 10 or 11, 64-bit | NVIDIA GPU with 8 GB+ VRAM |
| **Linux** | Ubuntu 22.04 / Debian 12, 64-bit | NVIDIA GPU with 8 GB+ VRAM |
| **Memory** | 8 GB (small models) | 16 GB runs 7-billion-parameter models in 4-bit |
| **Disk** | 10 GB free | 50 GB+ if you keep several models |

A GPU is optional. Without one, Noelle runs smaller models on the CPU and tells you up front which models would be
slow.

## macOS

1. Open `Noelle-1.0.0-macos-arm64.pkg` and follow the steps. Choose **Install for me only** if you don't have an
   administrator password.
2. Or open the `.dmg` and drag **Noelle** into **Applications**.
3. The first time you open Noelle, macOS may say it can't verify the developer. Open **System Settings ›
   Privacy & Security**, scroll down and click **Open Anyway**.

## Windows

1. Run `Noelle-1.0.0-windows-x64-setup.exe`.
2. If SmartScreen shows "Windows protected your PC", click **More info › Run anyway**.
3. The wizard lets you paste a Hugging Face token (optional) and creates a Start-menu shortcut.

## Linux

```bash
sudo apt install ./noelle_1.0.0_amd64.deb
```

Then open **Noelle AI Studio** from your app menu. The `.tar.gz` archive is a portable alternative: unpack it and
run `Noelle/Noelle`.

## The setup assistant

On first launch the setup assistant walks you through three steps. You can run it again any time from
**Settings › About** or with `noelle setup`.

1. **Hugging Face token** (optional) — needed only for gated models such as some Llama releases. It is stored in
   your system keychain (Keychain on macOS, Credential Manager on Windows, Secret Service on Linux), never in a file.
2. **Hardware check** — what your GPU and memory can run.
3. **Optional downloads** — the llama.cpp tools for GGUF models and a small starter chat model. Nothing is
   downloaded unless you tick it.

## Uninstall

- **macOS:** delete **Noelle** from Applications (or `~/Applications`).
- **Windows:** **Settings › Apps › Installed apps › Noelle › Uninstall**.
- **Linux:** `sudo apt remove noelle`.

Your settings, experiments and knowledge bases are in `~/.noelle`, and downloaded models are in the shared Hugging
Face cache (`~/.cache/huggingface`). Delete those folders too if you want to remove everything — see
[Storage](docs:storage).

## Run from source

```bash
git clone https://github.com/thirza258/NoelleApp
cd NoelleApp
python3 -m venv .venv
.venv/bin/pip install -e ".[all]"
.venv/bin/python -m noelle
```

Python 3.11 or newer is required. On NVIDIA systems, install the CUDA build of PyTorch first.
