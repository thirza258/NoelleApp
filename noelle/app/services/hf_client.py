"""Hugging Face Hub metadata client for the desktop shell.

* The access token lives in the OS credential store (macOS Keychain, Windows
  Credential Manager, Secret Service) via ``keyring``; it is only attached to Hub
  metadata calls and passed to download child processes through their
  environment. It is never written to settings, logs or prompts.
* All network calls run on the shared QThreadPool (see :mod:`task_runner`).
* Listing uses the Hub REST API with Link-header pagination so the discovery
  grid can load further pages on scroll without refetching earlier ones.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

from noelle.runtime import tasks
from noelle.runtime.preflight import params_from_safetensors, parse_param_count

KEYRING_SERVICE = "noelle-desktop-studio"
KEYRING_USER = "huggingface-token"
log = logging.getLogger("noelle.credentials")
hub_log = logging.getLogger("noelle.hub")

LIST_EXPAND = ["author", "downloads", "likes", "pipeline_tag", "library_name", "tags", "safetensors",
               "trendingScore", "lastModified", "createdAt", "gated"]
SORTS = {"trending": "trendingScore", "downloads": "downloads", "likes": "likes", "recent": "lastModified",
         "created": "createdAt"}
_QUANT_TAGS = {"gguf", "awq", "gptq", "4-bit", "8-bit", "bitsandbytes", "mlx", "exl2", "fp8", "compressed-tensors"}


# ----------------------------------------------------------------- token store
_backend_lock = threading.Lock()
_backend_ready = False
_backend_retry_at = 0.0
_BACKEND_RETRY_S = 30.0


def _backend_candidates() -> tuple[str, ...]:
    if sys.platform == "win32":
        return ("keyring.backends.Windows.WinVaultKeyring",)
    if sys.platform == "darwin":
        return ("keyring.backends.macOS.Keyring",)
    return ("keyring.backends.SecretService.Keyring", "keyring.backends.kwallet.DBusKeyring")


def _ensure_backend() -> None:
    """Use the OS vault even when keyring's plugin discovery did not run (some frozen/headless launches).

    The module-level ``keyring.get_password``/``set_password`` wrappers are deliberately kept: the
    resolved backend is installed with ``set_keyring``, so callers and tests see normal keyring APIs.
    """
    global _backend_ready, _backend_retry_at
    with _backend_lock:
        if _backend_ready or time.monotonic() < _backend_retry_at:
            return
        _backend_retry_at = time.monotonic() + _BACKEND_RETRY_S
        try:
            import keyring
            import keyring.core
            from keyring.backends.fail import Keyring as FailKeyring
            from keyring.backends.null import Keyring as NullKeyring

            current = keyring.get_keyring()
        except Exception as exc:  # noqa: BLE001 - keyring itself is broken; callers report the real error
            log.warning("Could not inspect the OS credential store: %s", exc)
            return
        if not isinstance(current, (FailKeyring, NullKeyring)):
            _backend_ready = True
            return
        for name in _backend_candidates():
            try:
                backend = keyring.core.load_keyring(name)
            except Exception as exc:  # noqa: BLE001 - try the next platform backend
                log.debug("Credential backend %s is unavailable: %s", name, exc)
                continue
            keyring.core.set_keyring(backend)
            _backend_ready = True
            log.info("Using the OS credential store (%s).", name)
            return
        log.warning("No usable OS credential store was found; Hugging Face tokens cannot be saved.")


def _reset_backend_cache() -> None:
    """Only for tests/diagnostics after the environment changes."""
    global _backend_ready, _backend_retry_at
    with _backend_lock:
        _backend_ready = False
        _backend_retry_at = 0.0


def get_token() -> str | None:
    try:
        _ensure_backend()
        import keyring

        return keyring.get_password(KEYRING_SERVICE, KEYRING_USER) or None
    except Exception as exc:  # noqa: BLE001 - discovery runs in contexts where no vault exists
        log.warning("Could not read the Hugging Face token from the credential store: %s", exc)
        return None


def set_token(token: str) -> None:
    """Store the token and verify it can be read back (a null backend must not look like success)."""
    token = token.strip()
    if not token:
        raise ValueError("The access token is empty.")
    _ensure_backend()
    import keyring

    try:
        keyring.set_password(KEYRING_SERVICE, KEYRING_USER, token)
        stored = keyring.get_password(KEYRING_SERVICE, KEYRING_USER)
    except Exception:
        _reset_backend_cache()
        raise
    if stored != token:
        _reset_backend_cache()
        raise RuntimeError(
            "The operating system credential store did not keep the token. "
            "On Linux, make sure a Secret Service (GNOME Keyring/KWallet) is running; "
            "on Windows/macOS, check that the app is allowed to use the credential store.")


def clear_token() -> None:
    try:
        _ensure_backend()
        import keyring

        keyring.delete_password(KEYRING_SERVICE, KEYRING_USER)
    except Exception as exc:  # noqa: BLE001 - deleting a missing credential is not an error
        log.debug("Could not remove the Hugging Face token: %s", exc)


# ------------------------------------------------------------------ data model
@dataclass
class ModelSummary:
    id: str
    author: str | None = None
    pipeline_tag: str | None = None
    library_name: str | None = None
    downloads: int = 0
    likes: int = 0
    trending_score: float | None = None
    tags: list[str] = field(default_factory=list)
    license: str | None = None
    param_count: int | None = None
    weight_bytes: int | None = None
    gated: bool | str = False
    created_at: datetime | None = None
    last_modified: datetime | None = None
    config: dict[str, Any] = field(default_factory=dict)
    card_data: dict[str, Any] = field(default_factory=dict)
    siblings: list[dict[str, Any]] = field(default_factory=list)
    gguf: dict[str, Any] | None = None
    base_models: list[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.id.split("/")[-1]

    @property
    def namespace(self) -> str:
        return self.author or (self.id.split("/")[0] if "/" in self.id else "")

    @property
    def family(self) -> str | None:
        return tasks.task_family(self.pipeline_tag)

    @property
    def modality(self) -> str | None:
        return tasks.modality_for_tag(self.pipeline_tag)

    @property
    def custom_code(self) -> bool:
        return "custom_code" in self.tags

    @property
    def prequantized(self) -> bool:
        if self.config.get("quantization_config"):
            return True
        return bool(_QUANT_TAGS.intersection(t.lower() for t in self.tags))

    @property
    def is_gguf(self) -> bool:
        return "gguf" in self.tags or (self.library_name == "gguf")

    @property
    def is_adapter(self) -> bool:
        return self.library_name == "peft" or any(t.startswith("base_model:adapter:") for t in self.tags)

    @property
    def architecture(self) -> str | None:
        archs = self.config.get("architectures")
        if isinstance(archs, list) and archs:
            return str(archs[0])
        return self.config.get("model_type")

    @property
    def context_length(self) -> int | None:
        from noelle.runtime.preflight import context_limit

        if self.gguf and self.gguf.get("context_length"):
            return int(self.gguf["context_length"])
        return context_limit(self.config)

    @classmethod
    def from_info(cls, info: Any) -> ModelSummary:
        tags = list(getattr(info, "tags", None) or [])
        card = getattr(info, "card_data", None) or getattr(info, "cardData", None)
        card_dict: dict[str, Any] = {}
        if card is not None:
            card_dict = card.to_dict() if hasattr(card, "to_dict") else dict(card)
        license_name = card_dict.get("license") or next((t.split(":", 1)[1] for t in tags if t.startswith("license:")),
                                                        None)
        if isinstance(license_name, list):
            license_name = ", ".join(map(str, license_name))
        params, weight_bytes = None, None
        st = getattr(info, "safetensors", None)
        if st is not None and getattr(st, "parameters", None):
            params, weight_bytes = params_from_safetensors(dict(st.parameters))
        gguf = getattr(info, "gguf", None)
        if params is None and gguf and gguf.get("total"):
            params = int(gguf["total"])
        if params is None:
            params = parse_param_count(info.id)
        siblings = []
        for s in getattr(info, "siblings", None) or []:
            size = getattr(s, "size", None)
            lfs = getattr(s, "lfs", None)
            if size is None and lfs is not None:
                size = getattr(lfs, "size", None)
            siblings.append({"name": s.rfilename, "size": size})
        base_models = [t.split(":", 2)[-1] for t in tags if t.startswith("base_model:")]
        return cls(
            id=info.id,
            author=getattr(info, "author", None),
            pipeline_tag=getattr(info, "pipeline_tag", None),
            library_name=getattr(info, "library_name", None),
            downloads=int(getattr(info, "downloads", 0) or 0),
            likes=int(getattr(info, "likes", 0) or 0),
            trending_score=getattr(info, "trending_score", None),
            tags=tags,
            license=license_name,
            param_count=params,
            weight_bytes=weight_bytes,
            gated=getattr(info, "gated", False) or False,
            created_at=getattr(info, "created_at", None),
            last_modified=getattr(info, "last_modified", None),
            config=dict(getattr(info, "config", None) or {}),
            card_data=card_dict,
            siblings=siblings,
            gguf=dict(gguf) if gguf else None,
            base_models=base_models,
        )


# ---------------------------------------------------------------------- pager
class ModelPager:
    """Cursor over ``/api/models`` that follows the ``Link: <...>; rel="next"`` header."""

    def __init__(self, *, pipeline_tags: tuple[str, ...] | None = None, search: str | None = None,
                 sort: str = "trending", page_size: int = 30, author: str | None = None,
                 token: str | None = None, endpoint: str | None = None) -> None:
        self.pipeline_tags = pipeline_tags or ()
        self.search = (search or "").strip() or None
        self.sort = SORTS.get(sort, sort)
        self.page_size = page_size
        self.author = author
        self.token = token
        self.endpoint = endpoint
        self._next_urls: dict[str | None, str | None] = {}
        self._started = False
        self._exhausted = False
        self._lock = threading.Lock()
        self._seen: set[str] = set()

    @property
    def exhausted(self) -> bool:
        return self._exhausted

    def _base_params(self, tag: str | None) -> dict[str, Any]:
        per_stream = self.page_size if len(self.pipeline_tags) <= 1 else max(10, self.page_size // len(self.pipeline_tags))
        params: dict[str, Any] = {"limit": per_stream, "expand": LIST_EXPAND, "sort": self.sort}
        if tag:
            params["pipeline_tag"] = tag
        if self.search:
            params["search"] = self.search
        if self.author:
            params["author"] = self.author
        return params

    def fetch_next(self) -> list[ModelSummary]:
        from huggingface_hub import constants
        from huggingface_hub.hf_api import ModelInfo
        from huggingface_hub.utils import build_hf_headers, get_session, hf_raise_for_status

        with self._lock:
            if self._exhausted:
                return []
            endpoint = (self.endpoint or constants.ENDPOINT).rstrip("/")
            headers = build_hf_headers(token=self.token)
            session = get_session()
            streams: list[str | None] = list(self.pipeline_tags) or [None]
            if not self._started:
                for tag in streams:
                    self._next_urls[tag] = "__first__"
                self._started = True
            results: list[ModelSummary] = []
            for tag in streams:
                url = self._next_urls.get(tag)
                if not url:
                    continue
                if url == "__first__":
                    response = session.get(f"{endpoint}/api/models", params=self._base_params(tag), headers=headers,
                                           timeout=20)
                else:
                    response = session.get(url, headers=headers, timeout=20)
                hf_raise_for_status(response)
                self._next_urls[tag] = response.links.get("next", {}).get("url")
                items = response.json()
                skipped = 0
                new_items = 0
                for item in items:
                    try:
                        summary = ModelSummary.from_info(ModelInfo(**item))
                    except Exception:
                        skipped += 1
                        continue
                    if summary.id not in self._seen:
                        self._seen.add(summary.id)
                        results.append(summary)
                        new_items += 1
                hub_log.debug("Hub page for %s returned %d item(s), %d new, %d malformed",
                              tag or "all", len(items), new_items, skipped)
            if all(not self._next_urls.get(tag) for tag in streams):
                self._exhausted = True
            if len(streams) > 1:  # merge several pipeline tags by the active sort key
                results.sort(key=_sort_key(self.sort), reverse=True)
            return results


def _sort_key(sort: str) -> Callable[[ModelSummary], Any]:
    if sort == "downloads":
        return lambda m: m.downloads
    if sort == "likes":
        return lambda m: m.likes
    if sort in ("lastModified", "createdAt"):
        return lambda m: (m.last_modified or m.created_at or datetime.min).timestamp() if (m.last_modified or m.created_at) else 0
    return lambda m: m.trending_score or 0


# ------------------------------------------------------------- client facade
class HFClient:
    """Blocking Hub calls; run them through :func:`task_runner.runner`."""

    def __init__(self, endpoint: str | None = None) -> None:
        self.endpoint = endpoint

    @property
    def token(self) -> str | None:
        return get_token()

    def _api(self) -> Any:
        from huggingface_hub import HfApi

        return HfApi(endpoint=self.endpoint, token=self.token)

    def pager(self, *, modality: str | None = None, pipeline_tag: str | None = None, search: str | None = None,
              sort: str = "trending", page_size: int = 30) -> ModelPager:
        if pipeline_tag:
            tags: tuple[str, ...] = (pipeline_tag,)
        elif modality and modality in tasks.MODALITY_BY_KEY:
            tags = tasks.MODALITY_BY_KEY[modality].pipeline_tags
        else:
            tags = ()
        return ModelPager(pipeline_tags=tags, search=search, sort=sort, page_size=page_size, token=self.token,
                          endpoint=self.endpoint)

    def trending(self, count: int = 5, pipeline_tag: str | None = "text-generation") -> list[ModelSummary]:
        pager = ModelPager(pipeline_tags=(pipeline_tag,) if pipeline_tag else (), sort="trending",
                           page_size=max(count * 2, 10), token=self.token, endpoint=self.endpoint)
        items = [m for m in pager.fetch_next() if tasks.is_supported(m.pipeline_tag)]
        return items[:count]

    def model_info(self, repo_id: str) -> ModelSummary:
        hub_log.debug("Hub model_info: %s", repo_id)
        try:
            info = self._api().model_info(repo_id, files_metadata=True)
        except Exception as exc:  # noqa: BLE001 - log the classification, re-raise for the caller's UI
            hub_log.warning("Hub model_info failed for %s: %s: %s", repo_id, type(exc).__name__, exc)
            raise
        return ModelSummary.from_info(info)

    def fetch_file_text(self, repo_id: str, filename: str, max_bytes: int = 2_000_000) -> str | None:
        """GET a small repository file without writing it into the HF cache.

        ``hf_hub_download`` would create a README-only snapshot folder in the cache,
        which would then look like a (broken) downloaded model.
        """
        from huggingface_hub import hf_hub_url
        from huggingface_hub.utils import build_hf_headers, get_session

        url = hf_hub_url(repo_id, filename, endpoint=self.endpoint)
        hub_log.debug("Hub file request: %s/%s (max %d bytes)", repo_id, filename, max_bytes)
        data = bytearray()
        try:
            # Stream and stop at max_bytes: a repository file can be arbitrarily large, and
            # buffering the whole body first would let one oversized README exhaust memory.
            with get_session().stream("GET", url, headers=build_hf_headers(token=self.token), timeout=20,
                                      follow_redirects=True) as response:
                if response.status_code != 200:
                    hub_log.debug("Hub file %s/%s returned HTTP %d", repo_id, filename, response.status_code)
                    return None
                for chunk in response.iter_bytes():
                    data.extend(chunk[: max_bytes - len(data)])
                    if len(data) >= max_bytes:
                        break
        except Exception as exc:  # noqa: BLE001 - optional metadata must not abort discovery
            hub_log.warning("Hub file %s/%s failed: %s: %s", repo_id, filename, type(exc).__name__, exc)
            return None
        return bytes(data).decode("utf-8", errors="replace")

    def fetch_config(self, repo_id: str) -> dict[str, Any]:
        import json

        text = self.fetch_file_text(repo_id, "config.json")
        if not text:
            return {}
        try:
            return json.loads(text)
        except ValueError:
            return {}

    def fetch_readme(self, repo_id: str) -> str:
        return self.fetch_file_text(repo_id, "README.md") or ""

    def whoami(self, token: str | None = None) -> dict[str, Any]:
        from huggingface_hub import HfApi

        info = HfApi(endpoint=self.endpoint).whoami(token=token or self.token)
        hub_log.debug("Hub whoami succeeded for %s", info.get("name") or info.get("fullname") or "(unknown)")
        return info
