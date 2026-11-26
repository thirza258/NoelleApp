"""Typed runtime failures mapped onto protocol error codes."""

from __future__ import annotations

import re

from noelle.runtime.protocol import ErrorCode

_SECRET_PATTERNS = (
    (re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"), "hf_****"),
    (re.compile(r"\bsk-(?:or-)?[A-Za-z0-9_-]{20,}\b"), "sk-****"),
    (re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{20,}"), "Bearer ****"),
)


def redact_secrets(text: str) -> str:
    """Mask credential-shaped strings before they reach logs, UI errors or crash reports."""
    for pattern, replacement in _SECRET_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


class NoelleRuntimeError(Exception):
    code: ErrorCode = ErrorCode.INTERNAL

    def __init__(self, message: str, hint: str | None = None, **extra: object) -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint
        self.extra = extra  # structured details for the UI, e.g. component="llama.cpp"


class ModelNotFound(NoelleRuntimeError):
    code = ErrorCode.MODEL_NOT_FOUND


class LoadFailed(NoelleRuntimeError):
    code = ErrorCode.LOAD_FAILED


class BadRequest(NoelleRuntimeError):
    code = ErrorCode.BAD_REQUEST


class Unsupported(NoelleRuntimeError):
    code = ErrorCode.UNSUPPORTED


class NeedsNewerRuntime(Unsupported):
    code = ErrorCode.NEEDS_NEWER_RUNTIME


class Gated(NoelleRuntimeError):
    code = ErrorCode.GATED


class DependencyMissing(NoelleRuntimeError):
    code = ErrorCode.DEPENDENCY_MISSING


class RemoteCodeBlocked(NoelleRuntimeError):
    code = ErrorCode.REMOTE_CODE_BLOCKED


class Cancelled(NoelleRuntimeError):
    code = ErrorCode.CANCELLED


class PreflightRejected(NoelleRuntimeError):
    """The memory estimate says the request cannot fit; refused before allocating."""

    code = ErrorCode.OOM


def require(module: str, extra: str) -> None:
    """Raise :class:`DependencyMissing` with an install hint if ``module`` is absent."""
    import importlib.util

    if importlib.util.find_spec(module) is None:
        raise DependencyMissing(f"Python package '{module}' is not installed in the runtime environment.",
                                hint=f'Install it with: pip install "noelle[{extra}]"')
