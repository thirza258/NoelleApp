# Fix plan: models that fail to run, or fail without a visible error (all model families)

Status: **analysis only — nothing implemented.** Branch base: `main` @ `725b7f0` (v1.1.1).
Date: 2026-11-25. Source: user report (Kokoro-82M, OmniVoice, Stable Audio 3) and the user's `app.log` (Windows build).
Scope: every family Noelle runs — **Text, Vision & multimodal (incl. detection/classification), Audio & speech (ASR + TTS), Diffusion & synthesis, Embeddings** — so the same class of error cannot recur in any of them.

## 1. Symptoms reported

| # | Model / action | What the user sees |
|---|----------------|--------------------|
| S1 | `hexgrad/Kokoro-82M` → Synthesize | "It has an error but I can't see the error" |
| S2 | `k2-fsa/OmniVoice` → load | `The checkpoint you are trying to load has model type 'omnivoice' but Transformers does not recognize this architecture` |
| S3 | `stabilityai/stable-audio-3-small-sfx` → Download | Download fails repeatedly, 14 × `403 Client Error … Cannot access gated repo`, offered "Retry to resume" |
| S4 | Any download finishing | `CRITICAL … main_window.py:149 toast → RuntimeError: Internal C++ object (Toast) already deleted` |
| S5 | Any download finishing | `CRITICAL … optimize_panel.py:236 _refresh_adapters → RuntimeError: Internal C++ object (QListWidget) already deleted`. Fires once per previously opened model. |

The audio reports are examples of four general failure classes. Each class also exists in the other families (section 2.6).

| Class | Meaning | Families affected |
|-------|---------|-------------------|
| F1 Invisible error | Error reached the UI but was clipped, swallowed or crashed the toast | all |
| F2 Unrunnable checkpoint offered as runnable | Model needs a library/layout Noelle has no engine for | all |
| F3 Raw library error instead of an actionable code | `LOAD_FAILED` + a transformers/diffusers traceback text with a snapshot path | all |
| F4 Access problem shown as transient | Gated repo → "Retry to resume" loop | all (download path is shared) |

## 2. Root causes (verified)

### RC1 — Toast system breaks after the first toast (hides every error) — P0, all families
- `noelle/app/ui/main_window.py:148-153` filters old toasts with `t.isVisible()`.
- `Toast._dismiss()` (`noelle/app/ui/widgets/common.py:500-503`) calls `deleteLater()` when it fades out.
- After the first toast is gone, `isVisible()` on its stale Python wrapper raises `RuntimeError`, so **every later `ctx.toast(...)` call crashes**. That covers download-failed toasts (`main_window.py:249`), mic errors and load messages.
- Reproduced offscreen: show a toast, dismiss it, flush `DeferredDelete`, call `toast()` again → `RuntimeError: libshiboken: Internal C++ object (Toast) already deleted.`
- Effect: in the user's log the "Download failed: …" toast for Stable Audio 3 never appeared.

### RC2 — Error lines clipped off-screen in most playgrounds (S1) — P0
- The worker delivers the Kokoro error correctly. Reproduced by running the real worker and sending `SYNTHESIZE_SPEECH` against a stand-in folder whose `config.json` mimics Kokoro's (no `model_type`), not the real Kokoro snapshot:
  `LOAD_FAILED: Could not load …: Unrecognized model in <snapshot path>. Should have a 'model_type' key in its config.json.`
- Status labels are built with `label("", "faint")`: no word wrap, not selectable, 12 px faint text.
- The message holds the full HF snapshot path (`C:\Users\…\models--hexgrad--Kokoro-82M\snapshots\<40-hex>`). It is one line about 1,400 px wide.
- Rendered at 1366 px window width: the text is cut right after the snapshot hash. The actual reason is off-screen and cannot be copied.
- Messages are put into rich text (`<span style=…>{message}</span>`) without `html.escape`, so any `<…>` in an exception disappears.
- The worker side makes it worse: `noelle/runtime/model_manager.py:281` builds `LoadFailed(f"Could not load {spec.model_id}: {message}")`, with the raw library message that contains the snapshot path.

Error display, per family:

| Family | Where errors render | Wraps | Selectable | Escaped | Verdict |
|--------|---------------------|-------|------------|---------|---------|
| Text | chat bubble `chat_canvas.py:182` (`QTextBrowser`, wrap anywhere, markdown with `html: False`) | yes | yes | yes | OK; only the snapshot-path noise |
| Vision / detection / classification | `vision_playground.py:169` label, set at `:255` | **no** | **no** | **no** | broken |
| Audio – ASR | `audio_playground.py:112`, set at `:352` | **no** | **no** | **no** | broken |
| Audio – TTS | `audio_playground.py:255`, set at `:352` | **no** | **no** | **no** | broken (S1) |
| Diffusion | `diffusion_playground.py:144`, set at `:258` | **no** | **no** | **no** | broken |
| Embeddings | `embedding_playground.py:63`, set at `:108` | **no** | **no** | **no** | broken |
| Load button (all families) | `model_detail_page.py:211`, set at `:611` | yes | no | **no** | partly |
| Other | `evaluate_page.py:316`, `component_dialog.py:78,85,134` | varies | no | **no** | partly |

### RC3 — Dead per-model panels still receive download signals (S5) — P1, all families
- `model_detail_page.py:387-417` (`_build_tabs`) rebuilds `OptimizePanel` and `KnowledgePanel` each time any model opens, and `deleteLater()`s the old ones.
- `optimize_panel.py:100` and `knowledge_panel.py:119` connect the long-lived `ctx.downloads.job_finished` to a **lambda**. PySide only auto-disconnects bound methods of a destroyed `QObject`. A lambda keeps firing into the deleted widget, so one CRITICAL appears per model opened earlier (5 in the log). It is also a memory leak.

### RC4 — Gated repo shown as a transient failure (S3) — P1, all families
- A valid token that has gated-repo read permission still gets `GatedRepoError` from `auth_check("stabilityai/stable-audio-3-small-sfx")` (checked read-only).
- So the cause is that **the account has not accepted or been approved for the model's license on its Hub page**. It is not a token-permission problem.
- `noelle/runtime/downloader.py:266-271` only handles `GatedRepoError` from `model_info()`. But `model_info()` succeeds for gated repos (metadata is public).
- Every file then 403s. `_is_retryable` (`downloader.py:183`) correctly does not retry, but the run ends with code `INCOMPLETE` and "Retry to resume" (`downloader.py:333-336`). The user retried 3 times, and got 14 × 403 each time.
- The existing GATED message says "add your Hugging Face token in Settings". That is wrong when `authenticated=True`, as in the log.
- No code in `noelle/app` handles a `GATED` code specially.
- Gated repos are common in every family: Llama/Gemma (text), PaliGemma (vision), Stable Diffusion 3 / FLUX dev (diffusion), Stable Audio (audio). This downloader path is shared by all of them.

### RC5 — Runnability is decided by task tag only (S1, S2, S3) — P1, all families
- `noelle/runtime/capabilities.py:349-360` (`_runtimes`) picks runtimes **only from the task family**, using `_FAMILY_RUNTIMES` (`capabilities.py:50-59`): TEXT/VISION/DETECTION/CLASSIFICATION/ASR/TTS → `transformers`, DIFFUSION → `diffusers`, EMBEDDING → `sentence-transformers`.
- `library_name`, file layout and config `model_type` are ignored, apart from gguf/mlx/exl2/aqlm/bitnet/onnx-only. So `ModelCapabilities.supported` is `True` for every model whose pipeline tag maps to a family (`tasks.py:20-38`).
- Discover has no filter for "runs in Noelle":
  - `discovery_page.py:39-44` `SPEC_FILTER_OPTIONS` covers memory fit only.
  - `model_card_grid.py:165-179` checks only `Compatibility.level`.
  - `discovery_page.py:231-236` filters by `tasks.is_supported(pipeline_tag)` only.
- `hf_client.py:30` already fetches `library_name` and `gated` in list results, so a library filter needs no extra Hub calls.
- The worker loaders assume one library per family and pass raw exceptions through as `LOAD_FAILED`:

| Family | Loader | Assumes |
|--------|--------|---------|
| Text | `generation.py:214` `TransformersTextEngine` (`AutoModelForCausalLM` / `AutoModelForSeq2SeqLM`); `generation.py:361` MLX; `generation.py:480/596` llama.cpp | known `model_type`, or a GGUF/MLX file |
| Vision (VLM) | `generation.py:777` `VisionLanguageEngine` (`AutoModelForImageTextToText` / `VisionEncoderDecoderModel`) | `model_type` in the image-text-to-text mapping, not just in `AutoConfig` |
| Detection / classification | `pipelines.py:167` `VisionPipelineEngine` (`transformers.pipeline(task)`) | transformers checkpoint |
| ASR | `pipelines.py:48` `AsrEngine` (`pipeline("automatic-speech-recognition")`) | transformers checkpoint |
| TTS | `pipelines.py:107` `TtsEngine` (`pipeline("text-to-audio")`) | transformers checkpoint |
| Diffusion | `pipelines.py:238` `DiffusionEngine` (`AutoPipelineForText2Image`, then `DiffusionPipeline`) | diffusers layout (`model_index.json`) |
| Embeddings | `pipelines.py:358` `EmbeddingEngine` (`SentenceTransformer`) | sentence-transformers or plain transformers encoder |

- Extra loader defect: `pipelines.py:248-251` swallows the `AutoPipelineForText2Image` exception and only reports the fallback's error, so the first (often clearer) reason is lost.

### 2.6 Unrunnable-checkpoint patterns per family (to encode as rules)
These are the shapes F2/F3 take in each family. The detection rules in Phase 3 are generic (library allowlist + layout + `model_type` check), so they catch new libraries too. The library names below are examples for hint text; check each one against the Hub when building the table.

| Family | Runnable today | Unrunnable patterns (examples) |
|--------|----------------|--------------------------------|
| Text | transformers causal/seq2seq; GGUF via llama.cpp; MLX on Apple | `model_type` newer than the bundled transformers; ONNX/OpenVINO/CTranslate2/MLC-only repos; EXL2/AQLM/BitNet (already excluded); PEFT adapter without its base; remote-code without consent (handled by consent flow) |
| Vision & multimodal | `AutoModelForImageTextToText` models, VisionEncoderDecoder, transformers detection/classification pipelines | `model_type` known to `AutoConfig` but not to the image-text-to-text auto class; non-transformers vision libraries (e.g. YOLO/`ultralytics`, `timm`-only, `open_clip`); video-only models |
| Audio – ASR | transformers ASR (Whisper, wav2vec2, etc.) | NeMo checkpoints (`.nemo`), CTranslate2 / faster-whisper, whisper.cpp ggml, SpeechBrain / ESPnet, diarization pipelines |
| Audio – TTS | transformers text-to-audio (SpeechT5, VITS/MMS-TTS, Bark, MusicGen) | **Kokoro** (`kokoro`, no `model_type`), **OmniVoice** (`omnivoice`), **Stable Audio 3** (`stable-audio-3`, gated), other own-package TTS (e.g. F5-TTS, StyleTTS2, XTTS/Coqui, Piper ONNX, Parler-TTS package) |
| Diffusion & synthesis | diffusers-layout repos (`model_index.json`) | single-file checkpoints (only `.safetensors`/`.ckpt`, no `model_index.json`); ComfyUI-only repos; non-diffusers libraries; gated SD3/FLUX-dev without approval |
| Embeddings | sentence-transformers repos, plain transformers encoders (mean pooling) | Model2Vec / static embeddings, ONNX-only (FastEmbed), multi-vector ColBERT-style (PyLate), decoder-LLM embedders needing remote code without consent, `model_type` unknown |

Card facts for the three reported models (from their Hub pages):

| Model | Runtime the card requires | Loads in `transformers`/`diffusers`? | Notes |
|-------|---------------------------|--------------------------------------|-------|
| `hexgrad/Kokoro-82M` | `pip install kokoro>=0.9.2` (+ `misaki` G2P), system `espeak-ng` | not stated / no (config has no `model_type`) | Apache-2.0, 24 kHz, 8 languages / 54 voices; **Indonesian not listed** |
| `k2-fsa/OmniVoice` | `pip install omnivoice`; card pins `torch==2.8.0` / `torchaudio==2.8.0` | no (`model_type: omnivoice` unknown) | 600+ languages; weights **CC-BY-NC (non-commercial)**; base Qwen3-0.6B |
| `stabilityai/stable-audio-3-small-sfx` | Stability `stable-audio-3` lib (or `stable-audio-tools`) | no | **Gated**, sound effects (not speech), English, Stability Community License + Gemma terms |

## 3. Fix plan

Order matters: Phase 1 makes every later error visible. Phases 1–4 apply to **all families**.

### Phase 1 — Error visibility everywhere (P0)

**1.1 Toast crash** — `main_window.py:148-153`
- Filter with `shiboken6.isValid(t) and t.isVisible()`.
- Also drop each toast from `_toasts` on its `destroyed` signal.
- Acceptance: unit test shows a toast, dismisses it, flushes `QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)`, and calls `toast()` again without raising.

**1.2 One shared error widget for every playground**
- Add a shared helper `error_html(message, hint=None)` in `widgets/common.py`. It applies `html.escape` to message and hint and returns the red span. Use it at every site in the RC2 table.
- Better still, an `ErrorLine` widget with:
  - wrap and selectable text;
  - a size policy that cannot widen the panel (QLabel wraps only at whitespace, and a path is one token);
  - a "Copy details" action;
  - an "Open logs" link.
- Use it in Vision, ASR, TTS, Diffusion and Embedding playgrounds and in the Load-button status. Text keeps its chat bubble.
- Worker: in `model_manager.py:281`, replace the snapshot path (`source`) with `spec.model_id`. Keep the full text and traceback in `models.log`.
- Acceptance: one parametrized offscreen render test per playground family at 1100 px and 1366 px with a long error. The full reason is visible, and the widget's `sizeHint().width()` ≤ panel width.

**1.3 Disconnect dead panels** — `optimize_panel.py:100`, `knowledge_panel.py:119`
- Replace the lambdas with bound methods (`self._on_download_finished`).
- Audit every `ctx.*.connect(lambda …)` on per-model widgets. Pages are long-lived and are fine.
- Add a lint/grep check in CI: no lambda connected to a `ctx.*` signal inside `noelle/app/ui/widgets/`.
- Acceptance: test builds each per-model panel, `deleteLater()`s it, flushes, emits `job_finished`, and gets no `RuntimeError`.

### Phase 2 — Gated downloads (P1, shared download path)
- `downloader.py` `run()`:
  - If `info.gated`, call `api.auth_check(repo_id, token=token)` before the plan.
  - On `GatedRepoError`, emit `code: "GATED"` with the repo URL and stop.
- Also, if any per-file failure is a `GatedRepoError`, end with `GATED`, not `INCOMPLETE`.
  - Keep `GatedRepoError` before `RepositoryNotFoundError` in every `except` chain, because it is a subclass.
- Message branches on `bool(token)`:
  - Token present: "Your Hugging Face account is not approved for this model yet. Open https://huggingface.co/<repo>, accept the license / request access, then download again."
  - No token: "This model is gated. Accept its license on the Hub and add your token in Settings."
- App side (download panel and toast):
  - On `GATED`, hide "Retry to resume" and show **Open model page** instead.
  - Show a **Gated** badge on model cards in every category using `ModelSummary.gated`.
- Acceptance: unit test with a mocked `HfApi` (gated + `auth_check` raising) gets one `GATED` event and zero `file_error` events.

### Phase 3 — One runnability check, shared by shell and worker (P1, all families)
Create `noelle/runtime/runnability.py`: pure Python, no torch, importable by the desktop shell and the worker. It returns one verdict per model:

| Verdict | Meaning | User-facing action |
|---------|---------|--------------------|
| `RUNNABLE` | An engine exists and the layout matches | — |
| `NEEDS_COMPONENT` | Engine exists but is not installed (llama.cpp, diffusers, sentence-transformers) | Install component (existing flow) |
| `NEEDS_NEWER_RUNTIME` | `model_type` unknown to the bundled transformers/diffusers version | "Update Noelle" / known in a newer version |
| `REMOTE_CODE` | `auto_map` present | Existing consent flow |
| `UNSUPPORTED_LIBRARY` | `library_name` or layout belongs to a library Noelle has no engine for | Show the library; offer cloud fallback (Phase 6) |
| `GATED_NO_ACCESS` | Gated and the account isn't approved | Open model page |
| `UNKNOWN` | Not enough metadata (list view without config) | Exact check on the detail page |

Rules, applied per family:
1. **Library allowlist** per family. One table maps `library_name` to an engine:
   - `transformers`, `sentence-transformers`, `diffusers`, `peft` (with base), `gguf`, `mlx`;
   - anything else gives `UNSUPPORTED_LIBRARY`;
   - missing `library_name` defers to rules 2–3.
2. **Layout check**:
   - Text: `config.json` + weights, or `.gguf`, or MLX.
   - Diffusion: `model_index.json` (single-file → unsupported).
   - Embedding: `config.json` (`modules.json` preferred).
   - Vision/Audio: `config.json` with `model_type`.
3. **`model_type` check against the family's exact auto class.** This runs **only when rule 2 resolved the engine to transformers or sentence-transformers**. GGUF, MLX and diffusers layouts skip it: they have no root `model_type`, and GGUF-only repos have no `config.json` at all.
   - Text: causal-LM/seq2seq mappings.
   - VLM: the image-text-to-text mapping.
   - Detection/classification/ASR/TTS: the matching pipeline mapping.
   - One verdict per case:
     - `auto_map` present → `REMOTE_CODE`.
     - Missing `model_type` (Kokoro) → `UNSUPPORTED_LIBRARY`.
     - Unknown `model_type` and `library_name == "transformers"` → `NEEDS_NEWER_RUNTIME` ("a newer Noelle/transformers may support it").
     - Unknown `model_type` with any other or no `library_name` (OmniVoice) → `UNSUPPORTED_LIBRARY`, with the hint "not a transformers model; a future transformers release might add it". Never "Update Noelle".
4. The shell has no transformers, so the worker publishes its runtime facts at startup: in `HELLO`, or a new `RUNTIME_CAPS` command with the transformers/diffusers/sentence-transformers versions plus the supported `model_type` names per family. The shell caches them, so Discover and the detail page use the **same** rules as the worker.

Worker side:
- `model_manager._load_engine` runs the check **before** any loader. A non-runnable verdict raises a typed error: `Unsupported` / `DependencyMissing` / `RemoteCodeBlocked`.
- Error-code contract: every load failure maps to a code (`UNSUPPORTED`, `NEEDS_NEWER_RUNTIME`, `DEPENDENCY_MISSING`, `GATED`, `OOM`, `REMOTE_CODE_BLOCKED`). `LOAD_FAILED` is the last resort, and it always carries the model id and a hint, never a snapshot path.
- Diffusion loader: keep both exceptions (`AutoPipelineForText2Image` and the `DiffusionPipeline` fallback) and report the more specific one.
- Acceptance: worker test with fixture dirs gets the right code for each family (see section 4).

### Phase 4 — Filter unrunnable models in Discover, every category (P1, user request)
- `capabilities._runtimes(...)` calls `runnability` instead of the family-only table.
- `supported` becomes true for `RUNNABLE`, `NEEDS_COMPONENT`, `REMOTE_CODE` and `UNKNOWN`. It is false only for `UNSUPPORTED_LIBRARY`, `NEEDS_NEWER_RUNTIME` and `GATED_NO_ACCESS`.
  - This keeps Mission (`mission.py:382`) from rejecting remote-code VLMs or list-view models with thin metadata. The consent flow and the detail-page exact check handle those.
- Discover (`discovery_page.py:39-44`, `model_card_grid.py:165`), for **Text, Vision & multimodal, Audio & speech, Diffusion & synthesis, Embeddings**:
  - Add a filter **"Runs in Noelle"**: on by default (decision 1), hides `UNSUPPORTED_LIBRARY` cards.
  - Add **"Hide gated"**, or show gated models with a lock badge.
  - Card badge shows the verdict, e.g. "Needs `kokoro` — not supported in Noelle", "Needs newer runtime", "Gated".
- Limit: list results carry `library_name`/`tags`/`gated` but no `config.json`. Repos with no `library_name` (Kokoro) show `UNKNOWN` in the list.
  - Detail page: when config is fetched, run the exact check. Show a warning **before the Download button** ("This model can't run in Noelle yet — download anyway?").
- Library import: run the same check on already-downloaded models (`library_import.py`). Flag the ones that will fail before the user clicks Load.
- Acceptance: unit tests on `capabilities.analyze` (section 4 matrix).

### Phase 5 — Native engines (needs a decision; not part of the bug fix)
Install like llama.cpp: as an optional component through `environment_manager` / component dialog, with user consent. Each new engine adds one row to the Phase 3 library table, so it moves from `UNSUPPORTED_LIBRARY` to `RUNNABLE` with no other change.
- **Kokoro** (recommended first for audio):
  - `KPipeline(lang_code=…, repo_id="hexgrad/Kokoro-82M")` with offline voices from the snapshot (`voices/*.pt`), 24 kHz.
  - Needs the `kokoro` + `misaki` packages and `espeak-ng` (on Windows, a separate installer or a bundled DLL).
  - No Indonesian voice. Show that in the UI.
- **OmniVoice**:
  - The card pins `torch==2.8.0`, which conflicts with the CUDA 13.2 wheels the Windows build ships (`6fcd6bb`). It likely needs its own environment.
  - Weights are CC-BY-NC: show a non-commercial notice.
- **Stable Audio 3**: gated, sound effects (not speech). Needs Stability's library and a text-to-audio playground. Lowest priority.
- Other families, by demand:
  - Text: ONNX/OpenVINO runtime.
  - Vision: YOLO/`ultralytics` detection.
  - ASR: faster-whisper / CTranslate2 (big speed win on CPU).
  - Diffusion: single-file checkpoint loading (`from_single_file`).
  - Embeddings: Model2Vec.

### Phase 6 — Cloud fallback for models that can't run (from the earlier runner analysis)
- On `UNSUPPORTED_LIBRARY` / `NEEDS_NEWER_RUNTIME` / OOM / missing dependency, offer a cloud run with the user's key. Map by family:
  - Text/Vision → OpenRouter or OpenAI;
  - ASR/TTS/Embeddings → OpenAI;
  - Diffusion → none of the current providers.
- Only with explicit opt-in, and never in Local-Only mode.
- Depends on the runner fixes: real worker wiring in `LocalProvider`, and fixing the fail-open consent gate in `noelle/workflow/runner.py` (`consent_given=True` default).
- Tracked separately; listed here because it is the long-term answer when no local engine exists.

## 4. Test plan (regression guard so this does not happen again)
- **Fixture matrix** `tests/fixtures/checkpoints/<family>/<case>/`: config/layout files only, no weights. One good and several bad cases per family, used by both the shell-side and worker-side tests:

| Family | Good case | Bad cases → expected verdict/code |
|--------|-----------|-----------------------------------|
| Text | causal-LM config (e.g. `qwen2`); GGUF-only repo with no `config.json` → `RUNNABLE`/`NEEDS_COMPONENT` | unknown `model_type` + `library_name=transformers` → `NEEDS_NEWER_RUNTIME`; `auto_map` → `REMOTE_CODE`; ONNX-only → `UNSUPPORTED_LIBRARY` |
| Vision | `AutoModelForImageTextToText` config | `model_type` in AutoConfig but not image-text-to-text → `UNSUPPORTED_LIBRARY`; `library_name=ultralytics` → `UNSUPPORTED_LIBRARY` |
| ASR | `whisper` config | `.nemo` only → `UNSUPPORTED_LIBRARY`; CTranslate2 layout → `UNSUPPORTED_LIBRARY` |
| TTS | `speecht5` / `vits` config | Kokoro (no `model_type`) → `UNSUPPORTED_LIBRARY`; OmniVoice (`model_type: omnivoice`, not a transformers library) → `UNSUPPORTED_LIBRARY`; `library_name=stable-audio-3` → `UNSUPPORTED_LIBRARY` |
| Diffusion | `model_index.json` (no root `model_type`, rule 3 skipped) → `RUNNABLE` | single `.safetensors`, no `model_index.json` → `UNSUPPORTED_LIBRARY` |
| Embeddings | sentence-transformers (`modules.json`) | Model2Vec layout → `UNSUPPORTED_LIBRARY`; unknown `model_type` + `library_name=sentence-transformers` → `NEEDS_NEWER_RUNTIME` |
| Any | — | gated + no access → `GATED_NO_ACCESS` (mocked `HfApi`) |

- Worker tests: send each family's command (`GENERATE_STREAM`, `VISION_GENERATE`, `DETECT_OBJECTS`, `CLASSIFY_IMAGE`, `TRANSCRIBE`, `SYNTHESIZE_SPEECH`, `GENERATE_IMAGE`, `EMBED`) with each bad fixture. Expect the typed code, the model id, a hint, and no snapshot path in `message`.
- UI tests:
  - toast-after-dismiss;
  - dead-panel signal;
  - one long-error render test per playground (section 1.2 acceptance).
- Nightly metadata canary (optional, no downloads):
  - Run `runnability` over the top N trending models in each Discover category.
  - Report the share of `UNSUPPORTED_LIBRARY` and any new `library_name` values.
  - This catches new ecosystems (the next "Kokoro") before users hit them.
- Existing suites: `pytest tests/unit tests/e2e`. They have not been run yet against `725b7f0` on this machine; run before and after.
- Manual on Windows build:
  - open 3+ models of different families, finish a download: no CRITICAL lines in `errors.log`;
  - Kokoro and OmniVoice show a readable `UNSUPPORTED` reason;
  - Stable Audio 3 shows "not approved — open model page" once, with no retry loop.

## 5. Decisions needed from the user
1. Phase 4 default: hide unrunnable models in every category, or show them with a "not supported" badge?
2. Phase 3.4: add the worker → shell `RUNTIME_CAPS` handshake (exact checks in Discover), or keep shell-side checks to library/layout only?
3. Phase 5: which native engines, in what order? Suggested: Kokoro first; faster-whisper and diffusers single-file next.
4. Phase 6 depends on the open runner / cloud-fallback decision.

## 6. User actions (no code)
- Stable Audio 3: open https://huggingface.co/stabilityai/stable-audio-3-small-sfx while logged in and accept the license / request access. Even with access, it will not run until Phase 5 adds its engine.
- All fixes need a new Windows build before they show up in the installed app.
