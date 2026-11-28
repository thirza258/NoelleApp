# Training Studio

Noelle can fine-tune a local text-generation model on your own data without sending the
dataset or the resulting model anywhere. Training runs in a dedicated child process so the
desktop shell and the inference worker stay responsive.

## Workflow

1. Open **Training** in the top navigation.
2. Under **New training run**, choose a base model from the library or a local folder.
3. Choose a dataset file. **Inspect** shows row counts, duplicates, token statistics and short
   previews; the preview is displayed locally only and is never written to logs or manifests.
4. Pick LoRA (default), QLoRA (4-bit, NVIDIA CUDA), full fine-tuning or continued pretraining.
5. Review the hardware plan. **Apply recommendation** uses a conservative configuration for the
   detected GPU. The plan reports estimated VRAM/RAM/disk, time, steps and a risk verdict.
6. **Start training**. The run appears under **Runs** with a loss chart, logs, checkpoints and a
   manifest. Stop saves a checkpoint after the current optimizer step; **Resume** continues from the
   latest checkpoint.
7. Completed full models are registered in the library and can be selected in **Evaluate**. LoRA
   adapters can be merged into a full checkpoint from the run page before evaluation.

Training uses the app's Python by default. To isolate packages, create an environment in
**Settings › Runtime environments** with the `training` profile (add `cuda-quant` for QLoRA) and
select it in the Training page.

## Dataset formats

| Format | Accepted rows |
| --- | --- |
| JSONL / JSON | ChatML (`messages`), ShareGPT (`conversations`), instruction/input/output, query/response, or `text` |
| CSV / Parquet | The same field names as columns |
| TXT / Markdown | Paragraphs become plain-text training samples |

Rows are normalized to ChatML messages or plain text. Invalid rows are reported and skipped,
duplicates are removed, and the data is shuffled with the configured seed before the train/validation
split. Empty training splits fail with a visible error.

## CLI

```bash
noelle train prepare --dataset data.jsonl --output prepared/
noelle train plan --config recipe.yaml
noelle train run --config recipe.yaml
noelle train doctor
noelle train list
noelle train status <run-id>
```

`--json` on `run`, `prepare`, `plan`, `list` and `status` emits machine-readable output. The training
worker itself is launched as `noelle train-worker --run-id <id> --config <path>` by the desktop app.

Recipes are YAML or JSON objects with `name`, `output_name`, `mode`, `model`, `dataset`, `lora`,
`training` and `seed` keys. **Save recipe** in the Training page writes one.

## Storage

Everything lives under the Noelle data root (`~/.noelle` by default):

- `training.db` — runs, metrics and checkpoint index
- `projects/training/<run-id>/` — config, prepared datasets, checkpoints, logs, final model, manifest
- `projects/training/<run-id>/exports/` — merged adapters

The manifest includes the dataset SHA-256, row counts, hardware, package versions and the exact
recipe. It never includes dataset samples.

## Privacy and safety

- Dataset inspection, planning, training and evaluation are local.
- No automatic upload to the Hugging Face Hub or any cloud service.
- API tokens are never stored in recipes, configs, logs or model metadata.
- Dataset samples are excluded from worker events, logs and manifests.
- The model's remote code is only enabled when you tick the trust option (or have trusted the
  repository in Settings).
- Only one training run is active at a time, to avoid competing for GPU memory.
- QLoRA requires an NVIDIA CUDA GPU and `bitsandbytes`; other hardware should use LoRA or full
  fine-tuning.

## MVP scope

Text-generation SFT is implemented. Preference optimization (DPO/KTO), MLX training,
synthetic dataset generation, distributed/multi-GPU training and autonomous training agents are
not part of this release.