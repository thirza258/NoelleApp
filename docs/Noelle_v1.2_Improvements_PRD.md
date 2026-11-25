# Product Requirements Document: Noelle v1.2–v1.4 Improvements

| | |
|---|---|
| Status | Draft for review — nothing implemented |
| Date | 2026-11-25 |
| Code base | `main` @ `725b7f0` (v1.1.1) |
| Related | `docs/FIX_PLAN_model_runnability.md` (PR thirza258/NoelleApp#3), `docs/Noelle_Hybrid_Workflow_PRD.md`, `docs/LOGGING.md` |
| Architecture reminder | PySide6 desktop shell (`noelle/app`) + out-of-process ZeroMQ worker (`noelle/runtime/worker_service.py`); SQLite registry (`noelle/app/services/model_registry.py`); data root `~/.noelle` (`noelle/paths.py`) |

---

## 1. Summary

Noelle already downloads, analyzes, runs, evaluates and serves open models locally. This PRD defines eleven improvements:
- four that close everyday gaps users notice in the first hour: chat history, voice, update notices, problem reports;
- four that widen what Noelle can run: native engines, Ollama/LM Studio import, cloud fallback, speculative decoding;
- three that grow reach: Bahasa Indonesia interface, tools in chat, webcam input.

Each feature below states the evidence in the current code, the user stories, the UX, the technical design down to files and protocol messages, the privacy rules, acceptance criteria and tests.

### 1.1 Goals
- Make the first-hour experience complete: conversations persist, errors are reportable, and users learn about updates.
- Let more models run locally, and when one cannot, give an honest, consented alternative.
- Serve Indonesian users first-class (UI language, Indonesian speech).
- Keep every new capability consistent with Noelle's principles (section 1.3).

### 1.2 Non-goals
- No account system, cloud sync or telemetry/analytics.
- No automatic installer updates (notify and link only, in this cycle).
- No autonomous agents that act on the file system or shell. Tools in chat are read-only and limited (F3).
- No mobile app.

### 1.3 Principles (apply to every feature)
1. **Local first.** Default behaviour never sends prompts, audio, images or documents off the machine.
2. **No silent egress.** Any network use is visible, labelled with its destination, and switchable off. Cloud model calls need explicit opt-in plus confirmation.
   - One global switch, `privacy.local_only` (new, default `false`), turns off every optional network feature in this PRD: web-search tool, update checks, cloud fallback.
   - Do **not** use the existing `runtime.offline` as a gate. It only sets `HF_HUB_OFFLINE` for the worker, and it defaults to `true` (`settings_dialog.py:215`, `runtime_client.py:181`).
3. **Errors are visible and actionable.** Typed error codes, readable messages, a hint and a path to report.
4. **Fits the existing architecture.** The shell talks to the worker only through `noelle/runtime/protocol.py` commands; heavy work stays in the worker; long tasks never block the GUI thread.
5. **Translatable from day one.** Every new UI string goes through the i18n helper (F5).

---

## 2. Prerequisites (must land first)

These are bug fixes, specified in `docs/FIX_PLAN_model_runnability.md` and the earlier runner analysis. Several features depend on them.

| ID | Prerequisite | Needed by |
|----|--------------|-----------|
| P1 | Error visibility: toast crash fix, shared `ErrorLine`, no snapshot paths (fix plan Phase 1) | all |
| P2 | Typed load verdicts/codes (`UNSUPPORTED_LIBRARY`, `NEEDS_NEWER_RUNTIME`, …) and the shared `runnability` module (fix plan Phase 3) | F8, F9 |
| P3 | Gated download handling (fix plan Phase 2) | F6 |
| P4 | Workflow runner: wire `LocalProvider` to the real worker, run the workflow off the GUI thread, fail-closed consent (`consent_given=False` default, CLI `--allow-cloud`) | F8 |

---

## 3. Prioritization and release plan

Value scale: H/M/L. Effort: S = under 1 week, M = 1–2 weeks, L = more (one developer).

| ID | Feature | Value | Effort | Depends on | Target |
|----|---------|-------|--------|------------|--------|
| F2 | Chat history + prompt library | H | S | P1 | v1.2 |
| F7 | Report a problem | H | S | P1 | v1.2 |
| F6 | Update notifications (app + models) | M | S | P3 | v1.2 |
| F4 | Import from Ollama / LM Studio | M | S | P2 (verdicts) | v1.2 |
| F1 | Voice chat mode | H | M | F2 (history), P1 | v1.2 |
| F11 | Webcam + clipboard input for vision | M | S | P1 | v1.3 |
| F5 | Bahasa Indonesia UI | H (ID market) | M | none (do the string wrapping early) | v1.3 |
| F3 | Tools in chat | H | M | F2 | v1.3 |
| F10 | Speculative decoding | M | M | none | v1.3 |
| F9 | Native engines (Kokoro → faster-whisper → single-file diffusion → Model2Vec) | H | M each | P2 | v1.3–v1.4 |
| F8 | Cloud fallback | M | M | P2, P4 | v1.4 |

Release themes:
- **v1.2 "Daily driver"**: P1–P3, F2, F7, F6, F4, F1.
- **v1.3 "Reach"**: F5, F3, F10, F11, Kokoro + faster-whisper engines.
- **v1.4 "Everything runs"**: remaining F9 engines, F8 (after P4).

---

## 4. Feature specifications

### F1 — Voice chat mode

**Problem.** Noelle runs speech recognition, text generation and speech synthesis, but only as three separate playgrounds. Users cannot talk to a model.

**Evidence in code**
- Microphone capture exists: `MicRecorder` (`noelle/app/ui/widgets/audio_visualizer.py:74`). `stop()` returns a WAV path. Permission is requested via `request_microphone` (`:168`).
- Worker commands exist: `TRANSCRIBE`, `GENERATE_STREAM`, `SYNTHESIZE_SPEECH`, `STOP_GENERATION` (`noelle/runtime/protocol.py:42-48`).
- macOS mic permission string already exists (`packaging/noelle.spec:96`).
- The OpenAI-compatible server has `/v1/audio/transcriptions`, but **no `/v1/audio/speech`** (`noelle/runtime/server_service.py:9-12`).

**User stories**
- As a user, I hold a key (or click the mic), speak, and hear the model answer in a voice.
- As an Indonesian user, I speak Indonesian and hear Indonesian back.
- As a user, I can interrupt the model while it speaks by pressing the mic again.
- As a user, my voice conversation is saved like a text chat (F2), with the transcripts.

**UX**
- The Text playground gets a **Voice** toggle and a mic button next to Send.
- A voice settings popover offers:
  - ASR model picker (downloaded ASR models; default the smallest Whisper);
  - TTS model picker (downloaded runnable TTS models);
  - language (Auto / English / Indonesian / …);
  - voice/speaker;
  - speed;
  - "Auto-send when I stop talking" (silence detection) on/off.
- Push-to-talk: hold `Space` while the input box is not focused, or click to start and click to stop.
- A state indicator shows Listening → Transcribing → Thinking → Speaking, with a live waveform (`WaveformView`).
- Barge-in: pressing the mic while the model speaks stops playback and sends `STOP_GENERATION`.
- A memory banner appears when the three models together exceed the memory budget. It offers "Unload between turns (slower)" or a smaller model suggestion.

**Worker constraint (verified).** The worker runs **every model job on one GPU-lane thread, in arrival order** (`worker_service.py:10-18`, `_gpu_loop` at `:345`). Transformers generation runs inline on that thread with a callback streamer (`generation.py:284-300`). So the shell **cannot** overlap generation and speech by sending `SYNTHESIZE_SPEECH` while `GENERATE_STREAM` is running: the TTS request would queue until the whole answer is finished. Streaming speech must therefore happen **inside one worker job**.

**Technical design**
- New worker command `VOICE_TURN` (`protocol.py`), one job on the GPU lane. Payload:
  `{asr: spec, llm: spec, tts: spec, audio_path, language, messages, system_prompt, params, voice, speed}`.
  1. Transcribe `audio_path` with the ASR engine, then emit `TRANSCRIPT {text}`.
  2. Generate with the LLM engine, emitting `TOKEN_STREAM` as today. A **sentence chunker** inside `on_text` cuts a chunk at `.`, `!`, `?`, `…`, or a newline, once the chunk has 40 or more characters.
  3. Each chunk goes to a **TTS helper thread** owned by the job, which calls the TTS engine and emits `AUDIO_CHUNK {seq, text, audio_path, sampling_rate}`. Generation keeps running on the lane thread meanwhile.
     - The llama-server and llama-cpp engines generate outside PyTorch, so overlap is safe there.
     - For transformers or MLX generation on the same GPU, overlap means two threads using one device. **Verify thread safety on MPS and CUDA.** If it is unsafe, fall back to synthesizing each chunk between generation steps (still inside the job), or synthesize after generation (mode B below).
  4. Emit `RESULT {transcript, text, audio_chunks, metrics}`. `STOP_GENERATION` cancels generation and the TTS thread together (barge-in).
- Modes:
  - **A, streaming**: the default where overlap is safe.
  - **B, after generation**: always correct, slower.
  The job reports which mode it used in `RESULT.metrics`.
- New shell controller `noelle/app/services/voice_session.py` (`QObject`), with a state machine:
  `idle → recording → working (transcribing/thinking/speaking, driven by events) → idle` (any state → `idle` on cancel or error).
  - `MicRecorder.start()`. On stop (or silence detection: RMS below a threshold for 900 ms after speech), get `wav_path` and send one `VOICE_TURN`.
  - `TRANSCRIPT` adds the user bubble.
  - `TOKEN_STREAM` fills the assistant bubble.
  - Each `AUDIO_CHUNK` is appended, in `seq` order, to a playback queue (`QMediaPlayer` + `QAudioOutput`).
  - When playback drains after `RESULT`, return to `idle`. Delete the temporary WAVs, or keep them with the conversation if "Save audio" is on.
- Memory planning:
  - Before the first turn, estimate the three models with `noelle/runtime/preflight.py` (`estimate_runtime_bytes`, `plan_queue`) against `HardwareManager.memory_budget()`.
  - If the total exceeds the budget, choose a mode:
    - keep the LLM resident and load ASR/TTS on demand;
    - or block with a suggestion.
- Indonesian:
  - Whisper handles Indonesian ASR.
  - For TTS, recommend an Indonesian VITS checkpoint (for example `facebook/mms-tts-ind`; verify it is runnable via `runnability` before recommending).
  - Kokoro has no Indonesian voice (fix plan, card facts).
- API parity (optional in v1.2): add `POST /v1/audio/speech` to `server_service.py`, mapped to `SYNTHESIZE_SPEECH`, returning `audio/wav`.

**Privacy & security.** Fully local. The mic is only active while recording, shown by a red indicator. Audio files live under the data root and are deleted after playback unless "Save audio" is on.

**Acceptance criteria**
- On an Apple M2 Pro 16 GB (Whisper-small + a 0.5–3B instruct model + SpeechT5/VITS):
  - in mode A, first audio plays **≤ 3.0 s** after the user stops talking;
  - in mode B, first audio plays ≤ generation time + 1.5 s.
  The mode used is shown in the metrics row.
- Barge-in stops audio within 200 ms and generation within 500 ms.
- A conversation of 10 voice turns is saved and reopens with transcripts (F2).
- With any one of the three models failing to load, the user sees a typed error (P1/P2) naming the failing model.

**Tests**
- Unit:
  - sentence chunker (abbreviations such as "Dr.", decimals like "3.5", Indonesian punctuation);
  - state machine transitions with a fake runtime;
  - silence detection on synthetic audio.
- E2E (offscreen): a fake runtime replays recorded `TOKEN_STREAM`/`RESULT` events. The test asserts TTS requests are issued per sentence, in order.

**Risks / open questions**
- Echo: the speaker output is picked up by the mic. v1 is push-to-talk only by default; silence-detection auto-send is opt-in and headphones are recommended.
- Some TTS models are slow (real-time factor > 1). Show the RTF in voice settings and warn.

---

### F2 — Chat history and prompt library

**Problem.** Closing the app or pressing New chat loses the conversation. Only one global system prompt exists.

**Evidence in code**
- The Text playground keeps `self.history` in memory (`text_playground.py:250`).
- `new_chat()` clears it (`:218`).
- Saving is a manual JSON export (`export_chat`, `:326-336`).
- The single system prompt comes from `generation.system_prompt` (`default_settings.json`, `text_playground.py:129`).
- No conversation store exists anywhere in `noelle/`.

**User stories**
- My chats are saved automatically, and I can find them later by searching their text.
- I can rename, pin, delete and export a conversation.
- I can save system prompts as named presets and switch between them per chat.
- I can turn saving off (private chat) or delete everything.

**UX**
- A collapsible left sidebar in the Text playground:
  - search box;
  - "New chat";
  - list grouped by Today / Yesterday / Last 7 days / Older;
  - pinned chats at the top.
- Conversation title: auto-generated from the first user message (first 60 characters, trimmed to a word boundary). It can be renamed inline.
- The conversation header shows the model used and its parameters. Reopening a chat whose model is not downloaded shows a banner: "Model not downloaded — Download / Continue with current model".
- Prompt library:
  - a "Prompt" dropdown next to the System prompt toggle;
  - a "Manage prompts…" dialog (create, edit, duplicate, delete, import/export JSON).
- Built-in prompts (read-only, can be duplicated): Helpful assistant (current default), Penerjemah Indonesia ↔ English, Code reviewer, Summarizer, Guru (explains simply, in Indonesian).
- Variables in prompts: `{{date}}`, `{{time}}`, `{{language}}`. They are filled at send time.
- Private mode: an eye-off toggle. Messages are not written while it is on, and the chat disappears on close.
- Settings → Privacy: "Save chats" (default on), "Delete all chats…" (with confirmation).

**Data model.** Use a **separate database `~/.noelle/chats.db`**, with the same `sqlite3` connection pattern as `model_registry.py` and its own schema version in a `meta` table. User conversations then stay out of the model registry's quarantine-on-corruption path (`model_registry.py:244-285`), and "Delete all chats" can simply remove one file.

The registry does not set `foreign_keys` today, so `ON DELETE CASCADE` would silently do nothing. The external-content FTS table also needs triggers, or deleted messages keep matching searches. Both are part of the schema:

```sql
PRAGMA foreign_keys = ON;      -- run on every new connection, not just once
PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS conversations (
  id INTEGER PRIMARY KEY, title TEXT NOT NULL, model_id TEXT, adapter TEXT,
  system_prompt TEXT, prompt_id INTEGER, params_json TEXT,
  pinned INTEGER DEFAULT 0, created_at REAL NOT NULL, updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
  id INTEGER PRIMARY KEY, conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
  role TEXT NOT NULL, content TEXT NOT NULL, sources_json TEXT, metrics_json TEXT,
  audio_path TEXT, created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS prompts (
  id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE, content TEXT NOT NULL,
  tags TEXT, builtin INTEGER DEFAULT 0, updated_at REAL NOT NULL
);
-- Full-text search when the SQLite build has FTS5; otherwise fall back to LIKE.
CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(content, content='messages', content_rowid='id');
CREATE TRIGGER IF NOT EXISTS messages_ai AFTER INSERT ON messages BEGIN
  INSERT INTO messages_fts(rowid, content) VALUES (new.id, new.content);
END;
CREATE TRIGGER IF NOT EXISTS messages_ad AFTER DELETE ON messages BEGIN
  INSERT INTO messages_fts(messages_fts, rowid, content) VALUES ('delete', old.id, old.content);
END;
CREATE TRIGGER IF NOT EXISTS messages_au AFTER UPDATE ON messages BEGIN
  INSERT INTO messages_fts(messages_fts, rowid, content) VALUES ('delete', old.id, old.content);
  INSERT INTO messages_fts(rowid, content) VALUES (new.id, new.content);
END;
```

**Technical design**
- New service `noelle/app/services/conversation_store.py`:
  - CRUD;
  - search (FTS5 if available, detected once);
  - export (JSON and Markdown);
  - `delete_all()`.
- All writes happen off the GUI thread through `task_runner.runner().submit` (a small queue so message order is preserved).
- `TextPlayground` changes:
  - on `send()`, persist the user message;
  - on `_on_finished`, persist the assistant message with `sources`/metrics from the payload;
  - on `_on_failed`, persist nothing for the assistant.
- The Storage page shows the chat database size, with "Delete all chats".
- Migration: the existing `generation.system_prompt` becomes the built-in "Helpful assistant" prompt's default text, and the setting is kept for compatibility.

**Privacy & security.** Data stays in `~/.noelle`. No chat content is written to logs (consistent with `docs/LOGGING.md`). Private mode guarantees nothing is written: assert in tests that no row is inserted.

**Acceptance criteria**
- Restarting the app restores the last open conversation.
- Searching any word from a message finds its conversation in < 200 ms with 10,000 messages.
- Deleting a conversation removes its messages (cascade) and FTS rows.
- Private mode writes zero rows.

**Tests.** Unit tests for the store (CRUD, cascade, FTS fallback, migration from schema N−1). E2E tests: send → restart → history restored; private mode; prompt variables substituted.

---

### F3 — Tools in chat (web search, documents, calculator)

**Problem.** Noelle has web search and document retrieval, but in chat they are fixed toggles applied before generation (grounding). The model cannot decide to look something up, compute, or check the date.

**Evidence in code**
- Tool-calling support is already detected per model:
  - from tags (`capabilities.py:76`, `_TOOL_TAGS`);
  - from the chat template (`capabilities.py:200-205`, `tool_calling`).
- Existing capabilities: `WEB_SEARCH` (`worker_service.py:823`, SearXNG at `web_search.searxng_url`, default `http://127.0.0.1:8888`) and `RAG_QUERY` (`worker_service.py:792`).
- Generation renders `apply_chat_template(msgs, add_generation_prompt=True)` **without** a `tools=` argument (`generation.py:236-240`, `:373-377`).
- llama-server is started without `--jinja` (`generation.py:612`; no `jinja` string anywhere in `generation.py`). That flag is needed for template-driven tool calls.
- Noelle is an MCP **server** (`noelle/runtime/mcp_server.py`), not an MCP client.

**User stories**
- I ask "what's the weather in Jakarta / latest news on X" and the model searches, then answers with sources.
- I ask about my documents and the model searches my knowledge base itself.
- I ask for arithmetic and get an exact result instead of a guess.
- I see every tool call the model made, with its input and output, and I can turn tools off.

**UX**
- A "Tools" chip row in the Text playground, shown only when `capabilities.tool_calling` is true. Otherwise it is disabled, with the tooltip "This model wasn't trained for tool use".
- Toggles:
  - Web search (network badge);
  - Knowledge base (collection picker);
  - Calculator;
  - Date & time.
- Each tool call renders as a collapsible card inside the assistant message: tool name, arguments, result summary, and duration.
- Web search shows a one-time consent per session: "This chat may send search queries to <searxng_url>".

**Technical design**
- Tool registry in the worker, new module `noelle/runtime/tools.py`, with a JSON-schema spec per tool:

| Tool | Executes | Network |
|------|----------|---------|
| `web_search(query)` | existing web search engine | yes (SearXNG URL) |
| `knowledge_search(query, collection)` | existing RAG query | no |
| `calculator(expression)` | safe evaluator: Python `ast` with numeric ops only; no names, calls or attributes | no |
| `current_datetime(timezone?)` | stdlib `zoneinfo` | no |

- Generation loop in `_cmd_generate` (`worker_service.py:637`) when `payload.tools` is non-empty:
  1. Render tools per engine:
     - `TransformersTextEngine` and `MlxTextEngine`: `tokenizer.apply_chat_template(msgs, tools=specs, add_generation_prompt=True)`.
     - `LlamaServerEngine`: start llama-server with `--jinja` and pass OpenAI-style `tools` to its `/v1/chat/completions`.
     - `LlamaCppTextEngine` (llama-cpp-python): pass `tools` to `create_chat_completion`. This depends on the GGUF's chat format; **verify** with the bundled llama-cpp-python version, and fall back to "tools unavailable for this engine".
  2. Stream tokens. A **tool-call parser** detects the common formats:
     - Hermes/Qwen `<tool_call>{json}</tool_call>`;
     - Llama 3.1+ JSON `{"name":…, "parameters":…}`;
     - Mistral `[TOOL_CALLS][…]`;
     - llama-server native `tool_calls`.
     Text before a tool call streams normally.
  3. On a call, emit a new event `TOOL_CALL {id, name, arguments}`, execute it with a per-tool timeout (web 10 s, others 2 s), then emit `TOOL_RESULT {id, ok, summary, duration_ms}`. Append `{"role": "tool", "name": …, "content": result_json}` and continue generating.
  4. Stop after **5** tool rounds, on the stop event, or on a final answer.
- Protocol: add `Event.TOOL_CALL` and `Event.TOOL_RESULT` to `protocol.py`. Old clients ignore unknown events (verify `RuntimeClient` dispatch ignores them).
- The OpenAI-compatible server passes `tools`/`tool_choice` through on `/v1/chat/completions` (optional, v1.3+).
- Phase 2 (later): MCP **client**. Users add external MCP servers in Settings. Their tools appear with per-tool consent, and tools with side effects need confirmation for each call.

**Privacy & security**
- Only read-only tools in v1.
- Web search is the only network tool. It is disabled when `privacy.local_only` is on (section 5.1) and in Local-Only workflows.
- Tool output is treated as untrusted model input: it is never executed and never used as instructions for Noelle itself.
- The calculator rejects anything except numeric literals and operators.

**Acceptance criteria**
- With Qwen2.5-Instruct (transformers and GGUF) and Llama-3.2-3B-Instruct, asking "What is 1234*5678?" produces a calculator call and the exact answer.
- Knowledge-base questions call `knowledge_search` and show sources.
- Tool rounds are capped at 5. A malformed tool JSON yields a visible "tool call could not be parsed" note, not a crash.

**Tests.**
- Parser unit tests per format (including partial and streamed chunks).
- Calculator fuzz tests (reject `__import__`, names, attributes, huge exponents).
- Loop test with a scripted fake engine.
- Protocol back-compatibility test.

---

### F4 — Import models from Ollama and LM Studio

**Problem.** Many users already have GGUF models from Ollama or LM Studio. Noelle only finds models in the Hugging Face cache, so users re-download gigabytes.

**Evidence in code.** `library_import.discover_cached_models` scans the HF cache only (`noelle/app/services/library_import.py:124-153`). `ModelRecord` already supports `source="local"`, `engine`, `gguf_file` and `local_path` (`model_registry.py:144-163`).

**User stories**
- On first run, or from Storage, Noelle finds my Ollama/LM Studio models and lists them, ready to run.
- Imported models are referenced in place. Nothing is copied, and deleting them in Noelle never deletes the other app's files unless I explicitly choose to.

**UX**
- Storage page: a "Find models from other apps" button. The dialog lists found models with app, name, size, quantization and verdict (P2), with a checkbox per model; "Import selected".
- First-run setup assistant: an optional step when either app's folder exists.
- Imported cards show an "Ollama" / "LM Studio" source badge. "Remove from library" never deletes the source file. A separate "Delete file…" asks for confirmation and warns that the other app will lose the model.

**Technical design**
- New module `noelle/app/services/external_models.py` with scanners. Paths are best known at the time of writing; **verify each on all three OSes during implementation** and make them configurable in Settings → Storage.
  - **Ollama**:
    - root: `OLLAMA_MODELS` env var, else `~/.ollama/models` (Windows `%USERPROFILE%\.ollama\models`);
    - read manifests under `manifests/registry.ollama.ai/<namespace>/<model>/<tag>` (JSON);
    - the layer with `mediaType` `application/vnd.ollama.image.model` points to `blobs/sha256-<digest>`, which is a GGUF file without extension.
  - **LM Studio**: `~/.lmstudio/models/<publisher>/<repo>/*.gguf` (older installs: `~/.cache/lm-studio/models`).
- Metadata: read the GGUF header with the `gguf` package (already used for GGUF support): `general.architecture`, parameter count, context length, `tokenizer.chat_template`, quantization type. Fill in `param_count`, `architecture`, `metadata`.
- Registration:
  - `ModelRecord(repo_id="ollama/<model>:<tag>" | "lmstudio/<publisher>/<repo>/<file>", source="local", engine="gguf", local_path=<dir>, gguf_file=<file>, pipeline_tag="text-generation", metadata={"external": "ollama"|"lmstudio", "blob": path})`.
  - The Ollama blob has no `.gguf` extension; confirm the bundled llama.cpp loads it by path. If it does not, create a hard link or symlink named `*.gguf` inside `~/.noelle/models/external/`.
- Re-scan on Storage open. Mark records whose source file disappeared as "Missing".
- **Verify during implementation:**
  - Where GGUF parsing runs. `library_import` runs in the desktop shell, while the `gguf` package is only required in the worker (`generation.py:222`). Either add `gguf` to the shell's dependencies, or parse the header in the worker (new `INSPECT_GGUF` command).
  - Id safety. Ids such as `ollama/<model>:<tag>` must never reach Hub lookups, or cache-path code that assumes `org/name` (the detail page fetches Hub metadata). Gate on `source == "local"`. Store the display name separately, and use a path-safe id (for example `ollama--<model>--<tag>`).

**Privacy & security.** Read-only scanning of local folders. No network. The other app's files are never modified.

**Acceptance criteria**
- With Ollama and LM Studio each holding 2 models, the scan finds all 4 in < 2 s. Import registers them, and each loads and chats via llama.cpp.
- Removing an imported model from the library leaves the source file intact.

**Tests.** Fixture folders mimicking both layouts (tiny fake GGUF headers): manifest parsing, missing blob, `OLLAMA_MODELS` override, Windows path handling.

---

### F5 — Bahasa Indonesia user interface

**Problem.** Noelle targets Indonesian users (Indonesian TTS test sentences in the Hybrid PRD §8.2, Indonesian default `language: "id"` in `NodeRequest`), but the UI is English-only.

**Evidence in code.** No `QTranslator`, `tr()` or locale handling exists in `noelle/app`. A rough count finds about 395 literal UI strings passed to `label/button/setText/setPlaceholderText/setToolTip/QMessageBox`.

**User stories**
- I choose Bahasa Indonesia and the whole interface (menus, pages, dialogs, errors, toasts) is in Indonesian.
- Technical terms that have no common Indonesian form stay in English (GGUF, token, VRAM).

**UX**
- Settings → General → Language: System default / English / Bahasa Indonesia. A restart is required in v1, with a "Restart now" button.
- Layouts must tolerate text about 30 % longer (Indonesian is usually longer than English).

**Technical design**
- New `noelle/app/i18n.py`:
  - `tr(text, context="Noelle")` wrapping `QCoreApplication.translate`;
  - `install(app, lang)` loading `noelle/app/i18n/noelle_<lang>.qm`.
- Mechanical pass: wrap every user-visible string in `tr()`. Use `self.tr()` inside `QObject` subclasses where context matters.
- Workflow:
  1. `pyside6-lupdate` → `noelle/app/i18n/noelle_id.ts`.
  2. Translate (glossary in `docs/i18n_glossary.md`).
  3. `pyside6-lrelease` → `.qm`.
  4. Bundle the `.qm` files via `packaging/noelle.spec` `datas`.
- Errors: the worker keeps English messages plus typed codes (P2). The shell maps known codes to translated messages and shows the English detail under "Details".
- Number, size and date formatting through `QLocale` (for example `1,5 GB` in Indonesian).

**Acceptance criteria**
- 100 % of `.ts` entries finished for `id` (CI check fails on `type="unfinished"`).
- A pseudo-locale (strings padded by 40 %) renders every page at 1366×768 without clipped or overlapping text.

**Tests.** CI job: run `lupdate` and fail if new strings are not in `.ts`. Offscreen screenshot test per page in the pseudo-locale.

---

### F6 — Update notifications (app and models)

**Problem.** Users do not learn about new Noelle releases or newer model revisions.

**Evidence in code**
- Version: `noelle/__init__.py:8` (`1.1.1`).
- Releases: `.github/workflows/release.yml` publishes installers to a GitHub release on `v*` tags (`softprops/action-gh-release`). The landing page links SourceForge.
- The registry stores `revision` and `commit_hash` per model (`model_registry.py:150-151`).
- No update check exists.

**User stories**
- When a new Noelle version exists, I see an unobtrusive banner with release notes and a download link.
- When a downloaded model has a newer revision on the Hub, I see "Update available" and can update it.

**UX**
- A banner in the main window (existing `Banner` widget) with "What's new" (release notes) and Download (opens the release page). "Skip this version" and "Remind me later" are available.
- Storage / Downloaded: an "Update available" badge per model. "Update" downloads the new snapshot through the existing download manager; the old snapshot is kept until the new one completes.
- Settings → Updates: "Check for Noelle updates" (default asked at first run), "Check for model updates" (default on, daily).

**Technical design**
- New service `noelle/app/services/update_checker.py`, run off the GUI thread at most once per 24 h, with the last check time stored in `meta`.
- App check:
  - `GET https://api.github.com/repos/thirza258/NoelleApp/releases/latest` with a generic `User-Agent: Noelle/<version>` (no user identifiers);
  - compare semver; ignore drafts and pre-releases unless "Beta updates" is on.
- Model check: for each `source="hub"` record with `commit_hash`, compare `HfApi.model_info(repo_id, revision=record.revision or "main").sha` (batched, at most 4 concurrent, short timeout). Skip local/quantized/external records.
- Both checks are skipped when `privacy.local_only` is on (section 5.1). Settings then shows "Updates are not checked in Local-only mode".

**Privacy & security.** The app check sends only an HTTP request to GitHub. The model check sends repo ids the user already downloaded to the Hub; the existing token is used only when the repo is gated. No installer is downloaded or executed automatically.

**Acceptance criteria**
- A mocked newer release shows the banner once. "Skip this version" suppresses it until the next version.
- A mocked newer sha shows the badge. Update keeps the old snapshot until success, and a failed update leaves the model usable.

**Tests.** Unit tests for semver comparison, throttling, `privacy.local_only` gating and model sha comparison (mocked `HfApi`).

---

### F7 — "Report a problem"

**Problem.** When something fails, users paste raw log fragments into chats and issues. Reports lack versions, hardware and context, and logs can contain paths with usernames.

**Evidence in code**
- Logs: `~/.noelle/logs` with `app.log`, `errors.log`, `worker.log`, `models.log`, `ipc.log`, `downloads.log` (`docs/LOGGING.md`).
- Redaction helper: `redact_secrets` covers `hf_…`, `sk-…`/`sk-or-…` and `Bearer …` (`noelle/runtime/errors.py:9-20`).
- Hardware detection: `hardware_manager.detect_gpu` (`noelle/app/services/hardware_manager.py:67`).
- Settings → About has "Open logs".
- Unhandled exceptions are logged as `CRITICAL` only. The user sees nothing.

**User stories**
- I click "Report a problem", describe what happened, preview exactly what will be included, and save a zip to attach to a GitHub issue.
- When the app hits an unexpected error, I get a small notice with a "Report" button.

**UX**
- Settings → About → "Report a problem…" (next to the existing "Open logs"), plus a keyboard shortcut. The app has no menu bar today; add a Help menu entry if one is introduced.
- Dialog fields:
  - description;
  - "What were you doing?";
  - checkboxes: Logs (last 2 MB of each relevant log), System info, Noelle settings (redacted), Loaded models.
  - "Preview" shows the exact text to be included.
  - Buttons: "Save report…" (zip) and "Open GitHub issue". The issue opens the browser with a pre-filled title and template only; logs are never put in the URL.
- Global exception hook: a non-blocking toast "Something went wrong" with Report / Dismiss, throttled to one per minute.

**Technical design**
- New `noelle/app/services/diagnostics.py`, `build_report(options) -> Path`.
- Report contents:
  - versions: Noelle, Python, PySide6, plus torch/transformers/diffusers from the worker. Reuse the P2 `RUNTIME_CAPS`/`HELLO` data;
  - OS and CPU/RAM/GPU;
  - device and memory budget;
  - log tails;
  - redacted settings;
  - loaded models (`LIST_LOADED`).
- Redaction pipeline, applied to every text file:
  1. `redact_secrets`.
  2. Replace the home directory and username with `~` / `<user>`.
  3. Drop values of any setting key matching `token|key|secret|password`.
  4. Remove any email address.
- `sys.excepthook` and Qt message handler: route to the diagnostics toast (the shell currently only logs `CRITICAL`).

**Privacy & security.** The report is created locally and only shared by the user. Nothing is uploaded by Noelle.

**Acceptance criteria**
- A generated zip contains no string matching `hf_[A-Za-z0-9]{20,}`, `sk-`, the OS username or the home path (automated scan in tests).
- An exception raised in a slot shows the toast within 1 s, and the app keeps running.

**Tests.** Redaction unit tests with seeded secrets and paths (Windows and POSIX). Report build test. Exception-hook test.

---

### F8 — Cloud fallback for models that cannot run locally

**Problem.** Some models cannot run on a given machine: unsupported library, too large, missing component. Users with an OpenAI or OpenRouter key want a one-click alternative.

**Evidence in code**
- Provider adapters and the keyring store exist (`noelle/providers/`), and Settings already saves the OpenAI/OpenRouter keys (`settings_dialog.py:107-170`).
- Gaps:
  - `LocalProvider` is not wired to the worker (`registry.py:40`, `local_provider.py:107`);
  - the workflow runner's consent defaults to `True` (`noelle/workflow/runner.py`);
  - OpenRouter vision drops the image;
  - `fetch_live_models()` is never called.

**User stories**
- When a model shows "Can't run here" (P2 verdict), I see "Run with OpenRouter (your key)", with the cloud model, the estimated cost and the destination, and I must confirm.
- I can disable cloud fallback completely, and it never appears when `privacy.local_only` is on or in a Local-Only workflow.

**UX**
- On a typed failure (`UNSUPPORTED_LIBRARY`, `NEEDS_NEWER_RUNTIME`, OOM / `PreflightRejected`, `DEPENDENCY_MISSING`), the error line gets a secondary action "Run in the cloud instead…". It is only shown when a key exists and fallback is enabled.
- The confirmation dialog shows:
  - provider;
  - mapped cloud model (editable);
  - destination URL;
  - estimated cost (from `pricing.py`);
  - "Remember for this chat".
- Every cloud answer is labelled "Ran on OpenRouter · <model> · $0.0004".

**Technical design**
- Family → provider map:
  - Text/Vision → OpenRouter (or OpenAI);
  - ASR/TTS/Embeddings → OpenAI;
  - Diffusion → not offered.
- Model mapping:
  - OpenRouter `GET /models` entries include `hugging_face_id`. Use it to map the HF repo to the OpenRouter id (cached for 24 h).
  - With no match, suggest the closest model of a similar size from the cached list, editable by the user.
- Shell-side `CloudFallbackService` calls the provider adapters off the GUI thread. Results feed the same playground views.
- Settings: `cloud_fallback.enabled` (default **false**), `cloud_fallback.require_confirmation` (default true, can only be turned off per chat).
- Never offered when `privacy.local_only` is on or a workflow is Local-Only.

**Privacy & security**
- Explicit opt-in, plus confirmation that names the destination.
- Keys come only from the OS keyring and are never logged.
- Fix P4's fail-open consent first.

**Acceptance criteria**
- With fallback disabled, no cloud action is ever shown.
- With fallback enabled and a key, a Kokoro TTS failure offers OpenAI TTS. Confirming produces audio labelled with provider and cost.
- With `privacy.local_only` on, nothing is offered.

**Tests.** Mapping tests with a recorded `/models` payload. Consent tests (no call without confirmation). `privacy.local_only` gating tests.

---

### F9 — Native engines beyond transformers

**Problem.** Popular models need non-transformers runtimes (fix plan section 2.6). Today they fail.

**Engine plug-in contract.** All engines follow the existing `load()` / `release()` shape in `pipelines.py` and `generation.py`, plus:
- `can_load(meta) -> verdict`: registered in the P2 `runnability` table, so the Discover filter and the load preflight pick it up automatically.
- `component`: the optional dependency set, installed through `environment_manager` with the same consent dialog as llama.cpp (`component_dialog.py`).

**Engines, in order**

| # | Engine | Family | Packages / system deps | Notes |
|---|--------|--------|------------------------|-------|
| 1 | Kokoro | TTS | `kokoro>=0.9.2`, `misaki`; `espeak-ng` | 24 kHz; offline voices from snapshot `voices/*.pt`; 8 languages, no Indonesian; Apache-2.0 |
| 2 | faster-whisper | ASR | `faster-whisper` (CTranslate2) | Much faster on CPU; CTranslate2-format repos; int8 on CPU |
| 3 | Diffusers single-file | Diffusion | `diffusers` (`from_single_file`) | Loads `.safetensors` checkpoints without `model_index.json`; needs base-config inference (SD1.5 / SDXL / FLUX) |
| 4 | Model2Vec | Embeddings | `model2vec` | Static embeddings, very fast on CPU |
| later | OmniVoice | TTS | `omnivoice`; card pins `torch==2.8.0` | Needs its own environment (conflicts with CUDA 13.2 wheels, `6fcd6bb`); CC-BY-NC notice |
| later | Stable Audio 3 | Text-to-audio | Stability `stable-audio-3` | Gated; sound effects; separate playground |

**Acceptance per engine.**
- A fixture checkpoint moves from `UNSUPPORTED_LIBRARY` to `RUNNABLE` after installing the component.
- The playground runs end to end.
- The memory estimate is within ±25 % of measured.
- The component installs and uninstalls cleanly on Windows, macOS and Linux CI runners.

**Risks.** Windows system dependencies (espeak-ng). Installer size, so components stay optional downloads, never bundled.

---

### F10 — Speculative decoding (faster text generation)

**Problem.** Local generation speed is the main complaint on mid-range hardware. Speculative decoding gives 1.3–2× more tokens per second with identical output for greedy decoding.

**Evidence in code.** No speculative or assisted generation exists. llama-server is launched without draft options (`generation.py:612`). The transformers engine calls `generate` without `assistant_model`.

**User stories**
- I turn on "Speed up" and get faster answers with the same quality.
- Noelle suggests a compatible small draft model I already have, or one to download.

**UX**
- Param panel → "Speed-up":
  - Off;
  - Prompt lookup (no extra model; good for RAG and summaries);
  - Draft model, with a picker filtered to compatible models.
- Metrics row: tokens/s and draft acceptance rate after each answer.

**Technical design**
- transformers:
  - `model.generate(..., assistant_model=draft_model)` (assisted generation). The draft must share the tokenizer (check vocab size and tokenizer class).
  - Prompt lookup uses `prompt_lookup_num_tokens=10`.
- llama.cpp server: add draft arguments (`--model-draft <gguf>`, draft min/max). **Verify the flag names against the bundled llama-server build**; the build check already exists (`generation.py:608`).
- MLX: the `mlx_lm` draft-model option in `stream_generate` (verify the API of the installed version).
- Compatibility rule: same tokenizer family and `vocab_size`.
  - Auto-suggest within the same publisher family, for example Qwen2.5-0.5B-Instruct as draft for Qwen2.5-7B-Instruct.
- Memory: add the draft estimate in `preflight`. Refuse when the total exceeds the budget.

**Acceptance criteria**
- On a fixed 20-prompt set with greedy decoding, output is byte-identical with and without speculation.
- Median speed-up ≥ 1.3× on M2 Pro (MLX or GGUF) and on a CUDA runner.

**Tests.** Equivalence test with tiny models (CI-sized). Compatibility checker unit tests.

---

### F11 — Webcam and clipboard input for vision

**Problem.** The vision playground accepts only picked or dropped files.

**Evidence in code.** `vision_playground.py:185-207` uses `QFileDialog` and also accepts drops (`dropEvent` exists). There is no camera or paste. `packaging/noelle.spec` has `NSMicrophoneUsageDescription` (`:96`) but **no `NSCameraUsageDescription`**.

**User stories**
- I click "Camera", see a live preview, take a photo and ask the model about it.
- I paste a screenshot with Ctrl/Cmd+V.

**UX**
- Buttons next to Open: "Camera" (preview dialog with Capture / Retake / Use photo, plus a device picker) and "Paste".
- An optional "Live" mode (v1.4) re-captures every N seconds for detection models.

**Technical design**
- `QCamera` + `QMediaCaptureSession` + `QImageCapture` (QtMultimedia, already used for audio). Save the capture to a temporary PNG and set `image_path`.
- Paste: `QGuiApplication.clipboard().image()` → temporary PNG.
- Packaging:
  - add `NSCameraUsageDescription` to the macOS Info.plist in `packaging/noelle.spec`;
  - Windows camera privacy settings need no manifest change;
  - Linux needs QtMultimedia GStreamer/FFmpeg backends (verify in the release smoke test).

**Privacy & security.** The camera is active only while the preview dialog is open. Captures are temporary and deleted when replaced or on close.

**Acceptance criteria.** Capture → describe works with a VLM and with a detection model. Paste works with a screenshot. Denying camera permission shows a clear message with steps to enable it.

---

## 5. Cross-cutting requirements

### 5.1 Settings additions (`noelle/config/default_settings.json`)

```json
{
  "privacy": { "local_only": false },
  "ui": { "language": "system" },
  "chats": { "save": true, "save_audio": false },
  "voice": { "asr_model": null, "tts_model": null, "language": "auto", "auto_send_on_silence": false },
  "tools": { "enabled": [], "max_rounds": 5 },
  "library": { "external_sources": { "ollama": true, "lmstudio": true }, "ollama_path": null, "lmstudio_path": null },
  "updates": { "check_app": null, "check_models": true, "beta": false, "skipped_version": null },
  "cloud_fallback": { "enabled": false, "require_confirmation": true },
  "generation": { "speculative": { "mode": "off", "draft_model": null } }
}
```

`updates.check_app = null` means "ask at first run". `privacy.local_only = true` overrides `tools` (web search), `updates.*` and `cloud_fallback.*`. Settings → Privacy shows it as "Local-only mode: Noelle makes no optional network calls".

### 5.2 Database migrations
- Schema version is stored in the existing `meta` table.
- Each feature adds an idempotent migration step, with a test migrating from the previous version.
- The registry's existing quarantine-on-corruption behaviour (`model_registry.py:244-285`) must also cover the new tables.

### 5.3 Protocol changes (`noelle/runtime/protocol.py`)

| Change | Feature |
|--------|---------|
| `Event.TOOL_CALL`, `Event.TOOL_RESULT` | F3 |
| `HELLO` payload (or new `RUNTIME_CAPS`) with library versions and supported `model_type`s per family | P2, F7 |
| Optional `speculative` block in `GENERATE_STREAM` payload | F10 |
| New command `VOICE_TURN`; new events `TRANSCRIPT`, `AUDIO_CHUNK` (one job on the GPU lane, because the lane runs jobs one at a time) | F1 |
| Server endpoint `/v1/audio/speech` mapped to `SYNTHESIZE_SPEECH` | F1 (optional) |

Rule: the shell must ignore unknown events, and the worker must ignore unknown payload keys. Add a back-compatibility test.

### 5.4 Threading and performance
- No network or disk I/O on the GUI thread. Use `task_runner.runner().submit`.
- Performance budgets:
  - app start to interactive: unchanged ±5 %;
  - chat history sidebar opens in < 150 ms with 1,000 conversations;
  - external model scan: < 2 s.

### 5.5 Accessibility
- Every new control has an accessible name and a keyboard path:
  - push-to-talk via keyboard;
  - tool cards are focusable;
  - camera capture with `Enter`.
- Colour is never the only state signal (mic recording shows an icon and text).

### 5.6 Logging
- New subsystems log to dedicated files following `docs/LOGGING.md`: `voice.log`, `tools.log`, `updates.log`.
- Never log prompts, transcripts, tool results containing user content, audio or images.

---

## 6. Success measures

Noelle has no telemetry, by design, so success is measured with:
- acceptance criteria and automated tests per feature (section 4);
- benchmark scripts committed under `tests/perf/`: voice latency (F1), speculative speed-up (F10), history search (F2);
- GitHub issue trends: share of issues that include a diagnostics report (F7) and "can't run model" issues (P2/F9), reviewed each release.

---

## 7. Open decisions

1. F6: is the app update check on by default, or asked at first run (proposed: ask)?
2. F1: ship silence-detection auto-send in v1.2, or keep push-to-talk only?
3. F3: include web search in v1.3 (network), or start with local tools only?
4. F5: is professional translation review needed, or community review?
5. F9: confirm engine order (Kokoro → faster-whisper → single-file diffusion → Model2Vec).
6. F8: is OpenAI the default provider for ASR/TTS/embedding fallback, or should users pick per family?
