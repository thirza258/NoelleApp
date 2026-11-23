---
title: Storage
nav: Storage
description: See where disk space goes, clean up unused models and partial downloads, and move the model cache.
section: Using Noelle
order: 14
slug: storage
---

Models are large. The **Storage** page shows where the space goes and helps you get it back.

## The Storage Advisor

At the top of the page, the advisor sums up the Hugging Face cache:

- **Active models** — used in the last 90 days.
- **Not used for 90+ days** — candidates for deletion, largest first, each with a **Delete** button.
- **Partial downloads** and **old revisions** — safe to prune.
- **Local builds** — 4-bit copies made in the Optimize tab.

**Potential cleanup** is the space you would free by removing all of these.

## Clean-up tools

- **Prune partial downloads** removes unfinished download files.
- **Prune old revisions** removes model versions no branch or tag points to any more.
- Right-click a repository to delete it or one of its revisions.
- **Relocate cache…** moves the whole cache to another drive.
- **Add cached models** registers complete models that are already in the Hugging Face cache (for example
  downloaded by other tools).

## Where Noelle keeps things

| Location | Contents |
|---|---|
| `~/.noelle/` (or `NOELLE_HOME`) | Settings, the model registry and experiments, local builds (`models/`), runtime environments, knowledge bases (`vector_store/`), exports and logs |
| `~/.cache/huggingface/hub` (or `HF_HOME`) | Downloaded models, shared with other Hugging Face tools |
| System keychain | Your Hugging Face token |

## Runtime environments

Advanced users can run the model runtime in a separate Python environment (**Settings › Environments**), for
example to try a newer PyTorch. `noelle env export NAME -o noelle-env.yaml` writes a lockfile with every package
pinned, and `noelle env import noelle-env.yaml` rebuilds the same environment on another machine.
