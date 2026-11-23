---
title: Command line
nav: Command line
description: Reference for the noelle command — analyze, run, eval, mcp, download, env and more.
section: Developers
order: 22
slug: cli
---

The same `noelle` program that opens the app also has commands for scripts and CI. After installing the Linux
package it is on your path; on macOS use `/Applications/Noelle.app/Contents/MacOS/Noelle`, and on Windows the
`Noelle.exe` in your installation folder. From source, use `python -m noelle`.

| Command | Purpose |
|---|---|
| `noelle` | Open the app |
| `noelle setup` | Open the setup assistant |
| `noelle analyze REPO` | Compatibility score, run options and recommendation for this machine |
| `noelle run -m MODEL "prompt"` | Send one prompt to a local model and stream the answer |
| `noelle eval …` | Run the Evaluation Lab from the terminal |
| `noelle mcp` | Model Context Protocol server on stdio |
| `noelle download REPO` | Verified, resumable download into the Hugging Face cache |
| `noelle env …` | Manage separate runtime environments |
| `noelle server` | The API server on its own (normally started from the app) |

## noelle analyze

```bash
noelle analyze Qwen/Qwen2.5-7B-Instruct
noelle analyze Qwen/Qwen2.5-7B-Instruct --profile low_memory --json
```

Profiles: `balanced`, `quality`, `fast`, `low_memory`, `max_context`.

## noelle run

```bash
noelle run -m HuggingFaceTB/SmolLM2-360M-Instruct "Explain KV caching in two sentences." --metrics
cat notes.txt | noelle run -m HuggingFaceTB/SmolLM2-360M-Instruct --system "Summarise the text."
```

Options: `--system`, `--precision`, `--device`, `--max-new-tokens`, `--temperature`, `--metrics` (prints time to first
token and speed to stderr).

## noelle eval

```bash
noelle eval --list-datasets
noelle eval -m MODEL [-m MODEL …] -d builtin:NAME|FILE.jsonl [-d …] \
    [-t "accuracy>=0.8"] [-t "tok_s>=20"] [-t "ttft_ms<=1000"] \
    [--baseline previous.json --max-regression 0.1] \
    [--json results.json] [--markdown report.md] [--recipe recipe.yaml] [--no-save]
```

Exit codes: **0** all checks passed, **1** a threshold or baseline check failed, **2** an error. Metrics for
thresholds: `accuracy`, `score`, `tok_s`, `ttft_ms`, `ttft_ms_p95`, `latency_ms`, `errors`. See the
[Evaluation Lab](docs:evaluation) for dataset and recipe formats.

## noelle mcp

```bash
noelle mcp                  # serve MCP on stdin/stdout
noelle mcp --print-config   # JSON to paste into an MCP client
noelle mcp --api-url http://127.0.0.1:9000
```

See [MCP server](docs:mcp).

## noelle download

```bash
noelle download HuggingFaceTB/SmolLM2-360M-Instruct
noelle download bartowski/Qwen2.5-7B-Instruct-GGUF --include "*Q4_K_M.gguf"
```

Options: `--revision`, `--include` / `--exclude` (patterns), `--local-dir`, `--max-workers`, `--retries`,
`--dry-run`. Files are checked against their published checksums.

## noelle env

```bash
noelle env create lab --profile base --profile rag
noelle env list
noelle env verify lab
noelle env export lab -o noelle-env.yaml
noelle env import noelle-env.yaml --name lab-copy
noelle env remove lab
```

Profiles: `base`, `diffusion`, `rag`, `server`, `peft`, `mlx`, `cuda-quant`, `gguf`.
