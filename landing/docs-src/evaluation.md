---
title: Evaluation Lab
nav: Evaluation Lab
description: Compare local models on built-in or custom datasets, keep a reproducible history and gate CI on accuracy and speed.
section: Using Noelle
order: 12
slug: evaluation
---

The Evaluation Lab answers *which model is best for my work?* with numbers from your own machine.

## Run an evaluation

1. Open **Evaluate** and tick the models to compare (downloaded text models and local builds).
2. Pick a dataset, or click the upload icon to import your own.
3. Keep **temperature 0** for repeatable answers and click **Run evaluation**.

Models run one after another on the GPU, each loaded once. If they don't all fit in memory, Noelle unloads each
model before loading the next. The leaderboard fills in live; click a model to see every case with its output,
score and timing.

## Built-in datasets

| Dataset | Checks |
|---|---|
| `reasoning-math` | Short arithmetic word problems (numeric answer) |
| `json-output` | Valid JSON with the right keys, types and values |
| `instruction-following` | Format rules: one word, all caps, lists, number ranges |
| `sentiment` | Positive / negative / neutral classification |
| `coding-python` | Small Python functions, checked statically — generated code is never run |
| `indonesian-qa` | Indonesian general knowledge |
| `translation-en-id` | English → Indonesian translation |
| `chat-latency` | Everyday prompts, timing only |

## Your own datasets

A dataset is a JSONL file: an optional first line describing it, then one case per line.

```json
{"_dataset": {"name": "support", "title": "Support tickets", "description": "Routing questions"}}
{"id": "t1", "prompt": "Customer: my card was charged twice. Which team? Answer billing, tech or sales.", "expected": "billing", "scorer": "choice", "choices": ["billing", "tech", "sales"]}
{"id": "t2", "prompt": "What is 12 x 8? Answer with just the number.", "expected": 96, "scorer": "numeric"}
```

| Scorer | Passes when |
|---|---|
| `exact` | The answer equals `expected` (case and punctuation ignored) |
| `contains` / `contains_all` | Any / all of the `expected` phrases appear |
| `regex` | Every pattern in `expected` matches and none in `forbid` does |
| `numeric` | The last number in the answer equals `expected` (optional `tolerance`) |
| `choice` | The first of the `choices` mentioned is `expected` |
| `json` | The answer is JSON and passes `required`, `types`, `equals`, `patterns`, `items` or `value` checks |
| `none` | Not scored — timing only |

Cases can also use `messages` instead of `prompt`, a `system` prompt, `tags` and their own `max_new_tokens`.

## History and the performance matrix

Every run is saved as an **experiment** with the dataset fingerprint, generation settings, hardware and each
model's exact revision. Open old runs from **History** and export them as JSON or Markdown. The
**Performance matrix** combines all finished runs: median speed, time to first token, peak memory and accuracy per
model, build and device. Runs from Compare appear in the history too.

## Compare

For a quick look without a dataset, **Compare** runs one prompt on up to four models side by side and shows time to
first token, tokens per second, peak memory and load time.

## From the command line and in CI

```bash
noelle eval --list-datasets
noelle eval -m HuggingFaceTB/SmolLM2-360M-Instruct -d builtin:sentiment -d ./support.jsonl \
    -t "accuracy>=0.7" -t "tok_s>=20" --json results.json --markdown report.md
noelle eval --recipe nightly.yaml --baseline last-week.json --max-regression 0.1
```

`noelle eval` exits with code **1** when a threshold or the baseline comparison fails, so it can gate a CI
pipeline. A recipe bundles a whole run:

```yaml
name: nightly
models: [HuggingFaceTB/SmolLM2-360M-Instruct, local/SmolLM2-360M-Instruct-mlx-4bit]
datasets: [builtin:instruction-following, ./support.jsonl]
params: {max_new_tokens: 256, temperature: 0}
thresholds: [accuracy>=0.7, tok_s>=25]
```
