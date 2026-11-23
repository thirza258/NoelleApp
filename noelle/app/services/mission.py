"""Mission Mode: turn a goal ("a local Indonesian legal assistant") into a runnable local stack.

The planner is deterministic and explainable — no hidden LLM decides anything:

1. **Understand the goal** — task (chat, code, documents/RAG, translation, vision,
   speech, image generation, embeddings), languages, priorities (fast, quality, low
   memory, long context), licence needs and any hardware the user states
   ("16 GB RAM and an RTX 3060").
2. **Find candidates** on the Hugging Face Hub (plus models already downloaded).
3. **Score each one** with the capability registry and the runtime advisor for the
   target hardware, then rank by fit, size, language match, task match and popularity.
4. **Propose a plan** — components (LLM, embedding model, …), the recommended build of
   each, total memory, and the steps to get there: download → build → index → test →
   serve. Every step that downloads, converts or starts a server needs the user's OK.

Qt-free: the Mission page executes the steps; tests drive the planner with fake search.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from noelle.runtime import advisor, capabilities, tasks
from noelle.runtime.advisor import Assessment, HardwareProfile
from noelle.runtime.preflight import GB, MemoryBudget, format_bytes, format_params

TASK_LABELS = {"chat": "Chat assistant", "code": "Coding assistant", "rag": "Document Q&A (RAG)",
               "translation": "Translation", "vision": "Image understanding", "asr": "Speech to text",
               "tts": "Text to speech", "image": "Image generation", "embeddings": "Semantic search"}
_TASK_WORDS: list[tuple[str, tuple[str, ...]]] = [
    ("rag", ("document", "documents", "pdf", "docx", "knowledge base", "knowledge", "rag", "manual", "contract",
             "legal", "policy", "policies", "my files", "dokumen", "berkas", "arsip", "hukum", "peraturan",
             "notes", "wiki", "q&a over", "question answering over")),
    ("code", ("code", "coding", "programming", "developer", "python", "javascript", "typescript", "sql", "kode",
              "pemrograman", "debug")),
    ("asr", ("transcribe", "transcription", "speech to text", "speech-to-text", "meeting recording", "podcast",
             "dictation", "transkrip", "rekaman")),
    ("tts", ("text to speech", "text-to-speech", "read aloud", "narrate", "narration", "voice over", "voiceover")),
    ("image", ("generate image", "generate images", "image generation", "text-to-image", "illustration", "draw",
               "artwork", "product photo", "logo", "buat gambar")),
    ("vision", ("image", "images", "photo", "photos", "picture", "screenshot", "ocr", "receipt", "gambar", "foto")),
    ("translation", ("translate", "translation", "translator", "terjemah", "menerjemahkan")),
    ("embeddings", ("semantic search", "embedding", "embeddings", "similarity", "deduplicate", "clustering")),
]
LANGUAGES = {
    "id": ("indonesian", "indonesia", "bahasa indonesia", "bahasa"), "en": ("english", "inggris"),
    "zh": ("chinese", "mandarin"), "ja": ("japanese",), "ko": ("korean",), "es": ("spanish",), "fr": ("french",),
    "de": ("german",), "ar": ("arabic",), "hi": ("hindi",), "vi": ("vietnamese",), "th": ("thai",),
    "ms": ("malay", "malaysian"), "jv": ("javanese", "jawa"), "su": ("sundanese", "sunda"), "pt": ("portuguese",),
    "ru": ("russian",), "it": ("italian",), "nl": ("dutch",), "tr": ("turkish",), "pl": ("polish",),
}
_INDONESIAN_HINTS = ("saya", "untuk", "dengan", "yang", "dokumen", "buat", "bisa", "tolong", "aplikasi")
MULTILINGUAL_FAMILIES = ("qwen", "gemma", "llama-3", "llama3", "aya", "mistral-nemo", "sailor", "seallm", "sahabat",
                         "komodo", "bloom", "mt5", "nllb", "seamless", "whisper", "bge-m3", "multilingual", "e5")
PERMISSIVE = ("apache-2.0", "mit", "bsd", "bsd-3-clause", "bsd-2-clause", "cc-by-4.0", "openrail", "openrail++",
              "creativeml-openrail-m", "unlicense", "cc0-1.0", "isc")
GPU_VRAM_GB = {"RTX 5090": 32, "RTX 5080": 16, "RTX 5070 Ti": 16, "RTX 5070": 12, "RTX 4090": 24, "RTX 4080": 16,
               "RTX 4070 Ti": 12, "RTX 4070": 12, "RTX 4060 Ti": 8, "RTX 4060": 8, "RTX 3090": 24, "RTX 3080 Ti": 12,
               "RTX 3080": 10, "RTX 3070 Ti": 8, "RTX 3070": 8, "RTX 3060 Ti": 8, "RTX 3060": 12, "RTX 2080 Ti": 11,
               "RTX 2080": 8, "RTX 2070": 8, "RTX 2060": 6, "GTX 1080 Ti": 11, "A100": 80, "H100": 80, "L4": 24,
               "T4": 16, "A10G": 24, "A10": 24, "V100": 16, "L40S": 48}
EVAL_FOR_TASK = {"chat": "instruction-following", "code": "coding-python", "rag": "instruction-following",
                 "translation": "translation-en-id", "embeddings": None}
EMBEDDINGS_MULTILINGUAL = ("BAAI/bge-m3", "intfloat/multilingual-e5-small",
                           "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
EMBEDDINGS_ENGLISH = ("sentence-transformers/all-MiniLM-L6-v2", "BAAI/bge-small-en-v1.5")
ROLE_TITLES = {"llm": "Language model", "embedding": "Embedding model", "asr": "Speech recognition",
               "tts": "Speech synthesis", "image": "Image generator", "vision": "Vision-language model"}
# Families known to load with Noelle's engines, preferred over newer repos that may need custom code.
CURATED = {"llm": ("qwen2.5", "qwen3", "llama-3", "gemma", "mistral", "phi-", "smollm", "granite", "deepseek-r1-distill"),
           "asr": ("whisper",), "tts": ("mms-tts", "speecht5", "bark", "parler"),
           "image": ("sd-turbo", "sdxl-turbo", "stable-diffusion", "flux"), "vision": ("qwen2.5-vl", "qwen2-vl",
                                                                                     "smolvlm", "llava", "paligemma"),
           "embedding": ("bge", "e5", "minilm", "mpnet", "gte", "nomic")}
ROLE_PIPELINES = {"asr": "automatic-speech-recognition", "tts": "text-to-speech", "image": "text-to-image",
                  "vision": "image-text-to-text", "embedding": "sentence-similarity"}

Search = Callable[..., list[Any]]  # search(pipeline_tag=, search=, filter=, sort=, limit=) -> [ModelSummary]
Info = Callable[[str], Any]  # repo id -> ModelSummary


@dataclass
class Goal:
    text: str
    task: str
    languages: list[str]
    profile: str
    commercial: bool
    hardware_note: str | None = None

    @property
    def title(self) -> str:
        names = [LANGUAGES[code][0].title() if code in LANGUAGES else code.title() for code in self.languages]
        return f"{TASK_LABELS.get(self.task, self.task)} · {', '.join(names) if names else 'any language'}"


_IMAGE_GENERATION = re.compile(r"\b(generate|create|make|draw|design|render|buat|bikin)\b(\s+\w+){0,3}?\s+"
                               r"(images?|pictures?|art|artwork|illustrations?|logos?|photos?|gambar)\b")


def parse_goal(text: str) -> Goal:
    lowered = f" {text.lower()} "
    task = "chat"
    if _IMAGE_GENERATION.search(lowered):
        task = "image"
    else:
        for name, words in _TASK_WORDS:
            if any(_has_word(lowered, w) for w in words):
                task = name
                break
    languages = [code for code, names in LANGUAGES.items() if any(_has_word(lowered, n) for n in names)]
    if "multilingual" in lowered and "multilingual" not in languages:
        languages.append("multilingual")
    if not languages and sum(_has_word(lowered, w) for w in _INDONESIAN_HINTS) >= 2:
        languages = ["id"]
    profile = "balanced"
    if any(_has_word(lowered, w) for w in ("fast", "quick", "realtime", "real-time", "low latency", "cepat")):
        profile = "fast"
    elif any(_has_word(lowered, w) for w in ("best quality", "accurate", "high quality", "quality", "akurat")):
        profile = "quality"
    elif any(_has_word(lowered, w) for w in ("low memory", "lightweight", "small model", "tiny", "ringan")):
        profile = "low_memory"
    elif any(_has_word(lowered, w) for w in ("long document", "long documents", "long context", "whole book")):
        profile = "max_context"
    commercial = any(_has_word(lowered, w) for w in ("commercial", "business", "company", "product", "komersial"))
    return Goal(text=text.strip(), task=task, languages=languages, profile=profile, commercial=commercial)


def _has_word(text: str, word: str) -> bool:
    return re.search(rf"(?<![a-z0-9]){re.escape(word)}(?![a-z0-9])", text) is not None


def target_hardware(text: str, detected: HardwareProfile) -> tuple[HardwareProfile, str | None]:
    """Plan for hardware the user describes ("16 GB RAM and an RTX 3060"), else this machine."""
    upper = text.upper()
    vram_table = dict(GPU_VRAM_GB)
    gpu = next((name for name in sorted(vram_table, key=len, reverse=True)
                if re.search(rf"(?<![A-Z0-9]){re.escape(name)}(?![0-9A-Z])", upper)), None)
    ram = None
    for amount, kind in re.findall(r"(\d{1,3})\s*GB\s*(VRAM|RAM|UNIFIED|MEMORY)?", upper):
        if kind == "VRAM" and gpu is None:
            gpu = f"GPU with {amount} GB"
            vram_table[gpu] = int(amount)
        elif kind != "VRAM":
            ram = int(amount)
    headroom = detected.budget.headroom
    if gpu:
        vram = vram_table[gpu]
        system = int((ram or 32) * GB)
        budget = MemoryBudget(int(vram * GB), headroom, system, "cuda")
        name = gpu if gpu.startswith("GPU with") else f"NVIDIA GeForce {gpu}"
        backends = {"transformers": True, "bitsandbytes": True, "llama.cpp": True, "llama.cpp-tools": True,
                    "diffusers": True, "sentence-transformers": True}
        profile = HardwareProfile("cuda", name, budget, backends, advisor.memory_bandwidth(name, "cuda"))
        return profile, f"{name} ({vram} GB VRAM)" + (f", {ram} GB RAM" if ram else "")
    mac = re.search(r"\bM([1-5])\s*(PRO|MAX|ULTRA)?\b", upper)
    if mac and ram:
        name = f"Apple M{mac.group(1)}{' ' + mac.group(2).title() if mac.group(2) else ''} GPU"
        budget = MemoryBudget(int(ram * GB * 0.75), headroom, int(ram * GB), "mps")
        profile = HardwareProfile("mps", name, budget, dict(detected.backends), advisor.memory_bandwidth(name, "mps"))
        return profile, f"{name} with {ram} GB unified memory"
    if ram and "CPU" in upper:
        budget = MemoryBudget(int(ram * GB * 0.7), headroom, int(ram * GB), "cpu")
        return HardwareProfile("cpu", "CPU", budget, dict(detected.backends), None), f"CPU with {ram} GB RAM"
    return detected, None


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------


@dataclass
class Candidate:
    summary: Any
    assessment: Assessment
    rank: float
    reasons: list[str]
    downloaded: bool = False

    @property
    def model_id(self) -> str:
        return str(self.summary.id)


@dataclass
class Component:
    role: str
    candidate: Candidate | None
    alternatives: list[Candidate] = field(default_factory=list)
    note: str = ""

    @property
    def title(self) -> str:
        return ROLE_TITLES.get(self.role, self.role)


@dataclass
class Step:
    key: str  # download | build | install | knowledge | test | serve
    title: str
    detail: str
    approval: str | None = None
    role: str | None = None
    status: str = "pending"  # pending | running | done | skipped | failed | manual
    message: str = ""


@dataclass
class MissionPlan:
    goal: Goal
    hardware: HardwareProfile
    hardware_note: str | None
    components: list[Component]
    steps: list[Step]
    total_memory: int | None
    warnings: list[str]
    eval_dataset: str | None

    def component(self, role: str) -> Component | None:
        return next((c for c in self.components if c.role == role), None)

    def to_dict(self) -> dict[str, Any]:
        def cand(c: Candidate | None) -> dict[str, Any] | None:
            if c is None:
                return None
            rec = c.assessment.recommended
            return {"model_id": c.model_id, "score": c.assessment.score, "rank": round(c.rank, 3),
                    "build": rec.label if rec else None, "runtime": rec.runtime if rec else None,
                    "memory_bytes": rec.memory_bytes if rec else None, "downloaded": c.downloaded,
                    "reasons": c.reasons}
        return {
            "goal": {"text": self.goal.text, "task": self.goal.task, "languages": self.goal.languages,
                     "profile": self.goal.profile, "commercial": self.goal.commercial},
            "hardware": {"device": self.hardware.device, "name": self.hardware.name,
                         "usable_bytes": self.hardware.usable_bytes, "stated": self.hardware_note},
            "components": [{"role": c.role, "choice": cand(c.candidate),
                            "alternatives": [cand(a) for a in c.alternatives], "note": c.note}
                           for c in self.components],
            "steps": [{"key": s.key, "title": s.title, "detail": s.detail, "needs_approval": bool(s.approval),
                       "status": s.status} for s in self.steps],
            "total_memory_bytes": self.total_memory,
            "warnings": self.warnings,
            "evaluation_dataset": self.eval_dataset,
        }

    def to_markdown(self) -> str:
        lines = [f"# Mission plan — {self.goal.text}", "", f"**Goal:** {self.goal.title} · profile "
                 f"{advisor.PROFILES.get(self.goal.profile, self.goal.profile)}",
                 f"**Hardware:** {self.hardware_note or self.hardware.name} · {format_bytes(self.hardware.usable_bytes)} "
                 "usable", "", "## Recommended stack", ""]
        for component in self.components:
            c = component.candidate
            if c is None:
                lines.append(f"- **{component.title}:** none found — {component.note}")
                continue
            rec = c.assessment.recommended
            memory = f", ~{format_bytes(rec.memory_bytes)}" if rec and rec.memory_bytes else ""
            build = f"{rec.label} ({rec.runtime}{memory})" if rec else "—"
            lines.append(f"- **{component.title}:** `{c.model_id}` — {build}, score {c.assessment.score}/100")
            for reason in c.reasons:
                lines.append(f"  - {reason}")
        if self.goal.task == "rag":
            lines.append("- **Vector store:** SQLite + sqlite-vec (local)")
        lines += ["", f"**Estimated memory:** {format_bytes(self.total_memory)}", "", "## Steps", ""]
        lines += [f"{i}. {s.title} — {s.detail}" for i, s in enumerate(self.steps, 1)]
        if self.warnings:
            lines += ["", "## Warnings", ""] + [f"- {w}" for w in self.warnings]
        return "\n".join(lines) + "\n"


class Planner:
    def __init__(self, *, search: Search, info: Info | None, hardware: HardwareProfile,
                 local_models: Iterable[Any] = (), measured: Callable[[str], dict[str, Any]] | None = None) -> None:
        self.search = search
        self.info = info
        self.detected = hardware
        self.local = {getattr(r, "repo_id", ""): r for r in local_models if getattr(r, "exists_on_disk", lambda: True)()}
        self.measured = measured or (lambda _repo: {})

    def plan(self, text: str) -> MissionPlan:
        goal = parse_goal(text)
        hardware, note = target_hardware(text, self.detected)
        goal.hardware_note = note
        components: list[Component] = []
        warnings: list[str] = []
        if goal.task in ("chat", "code", "rag", "translation"):
            components.append(self._pick_llm(goal, hardware))
        if goal.task in ("rag", "embeddings"):
            components.append(self._pick_embedding(goal, hardware))
        if goal.task in ("asr", "tts", "image", "vision"):
            components.append(self._pick_role(goal.task, goal, hardware))
        total = 0
        for component in components:
            rec = component.candidate.assessment.recommended if component.candidate else None
            if component.candidate is None:
                warnings.append(f"No {component.title.lower()} fits this hardware. {component.note}".strip())
            elif rec and rec.memory_bytes:
                total += rec.memory_bytes
        if total and total > hardware.usable_bytes:
            warnings.append(f"All components together need ~{format_bytes(total)}, more than the "
                            f"{format_bytes(hardware.usable_bytes)} usable; Noelle will swap them in and out.")
        if note:
            warnings.append(f"Planned for the hardware you described ({note}), not this machine. Downloads and "
                            "tests still run here.")
        dataset = EVAL_FOR_TASK.get(goal.task)
        if goal.task in ("chat", "rag") and "id" in goal.languages:
            dataset = "indonesian-qa"
        if goal.task == "translation" and "id" not in goal.languages:
            dataset = "instruction-following"
        steps = build_steps(goal, components, dataset)
        return MissionPlan(goal, hardware, note, components, steps, total or None, warnings, dataset)

    # ----------------------------------------------------------- candidates
    def _pick_llm(self, goal: Goal, hw: HardwareProfile) -> Component:
        queries: list[dict[str, Any]] = [{"pipeline_tag": "text-generation", "search": "Instruct", "sort": "downloads",
                                          "limit": 60}]
        for lang in goal.languages:
            if lang not in ("en", "multilingual"):
                queries.append({"pipeline_tag": "text-generation", "filter": lang, "sort": "downloads", "limit": 40})
        if goal.task == "code":
            queries.insert(0, {"pipeline_tag": "text-generation", "search": "Coder", "sort": "downloads", "limit": 40})
        pool = self._gather(queries, tasks.TEXT)
        candidates = [c for c in (self._score(s, goal, hw, "llm") for s in pool if _instruction_tuned(s)) if c]
        return self._component("llm", candidates, "Try a smaller model or a machine with more memory.")

    def _pick_embedding(self, goal: Goal, hw: HardwareProfile) -> Component:
        multilingual = any(lang not in ("en",) for lang in goal.languages)
        preferred = EMBEDDINGS_MULTILINGUAL if multilingual else EMBEDDINGS_ENGLISH
        pool: list[Any] = []
        for repo in preferred:
            summary = self._summary(repo)
            if summary is not None:
                pool.append(summary)
        if not pool:
            pool = self._gather([{"pipeline_tag": "sentence-similarity", "sort": "downloads", "limit": 20}],
                                tasks.EMBEDDING)
        candidates = [c for c in (self._score(s, goal, hw, "embedding", order=i) for i, s in enumerate(pool)) if c]
        return self._component("embedding", candidates, "Set an embedding model in Settings › Grounding.")

    def _pick_role(self, role: str, goal: Goal, hw: HardwareProfile) -> Component:
        query: dict[str, Any] = {"pipeline_tag": ROLE_PIPELINES[role], "sort": "downloads", "limit": 40}
        if role == "tts" and goal.languages and goal.languages[0] not in ("en", "multilingual"):
            query["filter"] = goal.languages[0]
        family = tasks.task_family(ROLE_PIPELINES[role])
        pool = self._gather([query], family)
        candidates = [c for c in (self._score(s, goal, hw, role) for s in pool) if c]
        return self._component(role, candidates, "No supported model fits this hardware.")

    def _gather(self, queries: list[dict[str, Any]], family: str | None) -> list[Any]:
        seen: dict[str, Any] = {}
        for record in self.local.values():
            if (family is None or record.family == family) and record.source == "hub":
                summary = _summary_from_record(record)  # no network for models already on disk
                seen.setdefault(summary.id, summary)
        for query in queries:
            try:
                results = self.search(**query)
            except Exception:  # offline or rate limited: plan with what is available
                continue
            for summary in results:
                if family is not None and tasks.task_family(getattr(summary, "pipeline_tag", None)) != family:
                    continue
                seen.setdefault(summary.id, summary)
        return list(seen.values())

    def _summary(self, repo: str) -> Any:
        if self.info is None:
            return None
        try:
            return self.info(repo)
        except Exception:
            return None

    def _score(self, summary: Any, goal: Goal, hw: HardwareProfile, role: str, order: int = 0) -> Candidate | None:
        if getattr(summary, "is_adapter", False):
            return None
        cap = capabilities.analyze(summary)
        if not cap.supported and not cap.conversions:
            return None
        assessment = advisor.assess(cap, hw, config=getattr(summary, "config", None) or None, profile=goal.profile,
                                    measured=advisor.summarize_measurements(self.measured(cap.model_id)))
        rec = assessment.recommended
        if rec is None or rec.status == advisor.UNSUPPORTED:
            return None
        reasons: list[str] = []
        fit = assessment.score / 100.0
        params = cap.param_count or 0
        size = min(max((math.log10(params) - 8.0) / 2.8, 0.0), 1.0) if params else 0.3
        if goal.profile in ("fast", "low_memory"):
            size = max(0.0, 1.0 - abs(size - 0.35) * 1.5)  # favour capable small models (~1-3B), not the tiniest
        downloads = int(getattr(summary, "downloads", 0) or 0)
        popularity = min(math.log10(downloads + 1) / 7.0, 1.0)
        lang = 0.0
        name = cap.model_id.lower()
        wanted = [code for code in goal.languages if code not in ("en",)]
        if wanted:
            if any(code in cap.languages for code in wanted):
                lang = 1.0
                reasons.append(f"Model card lists {', '.join(c for c in wanted if c in cap.languages)}")
            elif any(family in name for family in MULTILINGUAL_FAMILIES):
                lang = 0.6
                reasons.append("Multilingual model family")
        task = 0.0
        coder = "coder" in name or "code" in name
        if goal.task == "code":
            task = 1.0 if coder else 0.0
            if coder:
                reasons.append("Trained for code")
        elif coder and role == "llm":
            task = -0.5
        bonus = 0.0
        downloaded = cap.model_id in self.local
        if downloaded:
            bonus += 0.15
            reasons.append("Already downloaded")
        bonus += {advisor.READY: 0.1, advisor.BUILD: 0.0, advisor.INSTALL: -0.2}.get(rec.status, 0.0)
        if goal.commercial and not _permissive(cap.license):
            bonus -= 0.5
            reasons.append(f"Licence '{cap.license or 'unknown'}' may restrict commercial use")
        if cap.gated:
            bonus -= 0.05
        if role == "embedding":
            bonus += max(0.0, 0.3 - 0.1 * order)  # keep the curated order unless the fit says otherwise
        if any(family in name for family in CURATED.get(role, ())):
            bonus += 0.2
        if cap.remote_code.required:
            bonus -= 0.15
            reasons.append("Needs the repository's own Python code to load")
        if role == "llm" and params and params < 1_000_000_000:
            bonus -= 0.25  # sub-1B chat models rarely make useful assistants
        rank = 0.35 * fit + 0.25 * size + 0.15 * popularity + 0.15 * lang + 0.10 * task + bonus
        if params:
            reasons.insert(0, f"{format_params(params)} parameters, score {assessment.score}/100 on this hardware")
        reasons.append(f"Runs as {rec.label} with {rec.runtime} (~{format_bytes(rec.memory_bytes)})"
                       if rec.memory_bytes else f"Runs as {rec.label} with {rec.runtime}")
        return Candidate(summary, assessment, rank, reasons, downloaded)

    @staticmethod
    def _component(role: str, candidates: list[Candidate], note: str) -> Component:
        candidates.sort(key=lambda c: c.rank, reverse=True)
        if not candidates:
            return Component(role, None, [], note)
        return Component(role, candidates[0], candidates[1:4], "")

def build_steps(goal: Goal, components: list[Component], dataset: str | None) -> list[Step]:
    """download → build/install → knowledge → test → serve, for the chosen components."""
    steps: list[Step] = []
    for component in components:
        c = component.candidate
        if c is None:
            continue
        if not c.downloaded:
            size = c.assessment.capability.weight_bytes
            rec = c.assessment.recommended
            gguf = f" ({rec.gguf_file})" if rec is not None and rec.gguf_file else ""
            steps.append(Step("download", f"Download {c.model_id}{gguf}",
                              f"{component.title}; about {format_bytes(size)}" if size else component.title,
                              approval=f"Download {c.model_id}{gguf}"
                                       + (f" (about {format_bytes(size)})" if size else "") + "?",
                              role=component.role))
        rec = c.assessment.recommended
        if rec is not None and rec.status == advisor.BUILD and not rec.local_variant:
            steps.append(Step("build", f"Build {rec.label}",
                              f"One-time local conversion of {c.model_id} (needs ~{format_bytes(rec.weight_bytes)} "
                              "of disk).", approval=f"Convert {c.model_id} to {rec.label}? This takes a few "
                                                    "minutes and runs locally.", role=component.role))
        elif rec is not None and rec.status == advisor.INSTALL:
            steps.append(Step("install", f"Install {rec.runtime}", rec.reason or "", role=component.role,
                              status="manual"))
    if goal.task == "rag":
        steps.append(Step("knowledge", "Create a knowledge base", "Pick the documents to index (PDF, DOCX, "
                          "Markdown, text). They stay on this machine.", role="embedding"))
    if dataset and any(c.role == "llm" and c.candidate for c in components):
        steps.append(Step("test", f"Test on '{dataset}'", "Runs the built-in dataset and saves the result in "
                          "the Evaluation Lab.", role="llm"))
    alias = {"code": "noelle-code", "rag": "noelle-chat", "translation": "noelle-chat"}.get(goal.task, "noelle-chat")
    if any(c.role == "llm" and c.candidate for c in components):
        steps.append(Step("serve", f"Serve as '{alias}'", "Adds an OpenAI-compatible alias on the local API "
                          "server (loopback only).", approval=f"Point the '{alias}' API alias at this model and "
                                                              "start the local API server?", role="llm"))
    return steps


def _instruction_tuned(summary: Any) -> bool:
    name = str(getattr(summary, "id", "")).lower()
    tags = {str(t).lower() for t in getattr(summary, "tags", None) or []}
    if any(word in name for word in ("instruct", "-it", "_it", "chat", "assistant")) or "conversational" in tags:
        return "base" not in name.split("/")[-1].split("-")
    return False


def _permissive(license_name: str | None) -> bool:
    if not license_name:
        return False
    return any(part.strip().lower() in PERMISSIVE for part in str(license_name).split(","))


def _summary_from_record(record: Any) -> Any:
    from noelle.app.context import summary_from_record

    return summary_from_record(record)


def hub_search(client: Any) -> Search:
    """Hub search with tag filters (language codes), returning ``ModelSummary`` objects."""

    def search(*, pipeline_tag: str | None = None, search: str | None = None, filter: str | None = None,
               sort: str = "downloads", limit: int = 40) -> list[Any]:
        from huggingface_hub import HfApi

        from noelle.app.services.hf_client import LIST_EXPAND, ModelSummary

        api = HfApi(endpoint=getattr(client, "endpoint", None), token=getattr(client, "token", None))
        items = api.list_models(pipeline_tag=pipeline_tag, search=search, filter=filter, sort=sort, limit=limit,
                                expand=LIST_EXPAND)
        results = []
        for item in items:
            try:
                results.append(ModelSummary.from_info(item))
            except Exception:
                continue
        return results

    return search
