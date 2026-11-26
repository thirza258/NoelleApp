"""Read-only chat tools and a tolerant tool-call parser.

No tool in this milestone writes files, runs shell commands or executes model output.
``calculator`` accepts numeric literals/operators only; ``current_datetime`` is stdlib;
knowledge/web search are injected by the worker. Tools with side effects can be added
later behind the confirmation flow, but are deliberately absent here.
"""

from __future__ import annotations

import ast
import json
import math
import operator
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Mapping

TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL | re.IGNORECASE)
MISTRAL_RE = re.compile(r"\[TOOL_CALLS\]\s*(\[.*?\])", re.DOTALL | re.IGNORECASE)
FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)
JSON_OBJECT_RE = re.compile(r"\{[^{}]*\"(?:name|tool)\"[^{}]*\}", re.DOTALL)


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    risk: str = "read"
    network: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "parameters": self.parameters,
                "risk": self.risk, "network": self.network}


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "arguments": self.arguments}


@dataclass
class ToolResult:
    name: str
    ok: bool
    summary: str
    data: Any = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "ok": self.ok, "summary": self.summary, "data": self.data, "error": self.error}

    def as_message(self) -> str:
        if not self.ok:
            return f"Tool {self.name} failed: {self.error}"
        return f"Tool {self.name} result: {self.summary}"


class ToolError(ValueError):
    pass


TOOLS: dict[str, ToolSpec] = {
    "calculator": ToolSpec(
        "calculator", "Evaluate an arithmetic expression exactly.",
        {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}),
    "current_datetime": ToolSpec(
        "current_datetime", "Return the current date and time, optionally in an IANA timezone.",
        {"type": "object", "properties": {"timezone": {"type": "string"}}, "required": []}),
    "knowledge_search": ToolSpec(
        "knowledge_search", "Search the user's local knowledge base for relevant passages.",
        {"type": "object", "properties": {"query": {"type": "string"}, "collection": {"type": "string"}},
         "required": ["query"]}),
    "web_search": ToolSpec(
        "web_search", "Search the web through the configured SearXNG instance.",
        {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        risk="read", network=True),
}

_BIN_OPS: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow,
}
_UNARY_OPS: dict[type[ast.unaryop], Callable[[Any], Any]] = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_MAX_EXPONENT = 12


def _safe_number(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_safe_number(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left, right = _safe_number(node.left), _safe_number(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > _MAX_EXPONENT:
            raise ToolError("Exponent is too large.")
        value = _BIN_OPS[type(node.op)](left, right)
        if not math.isfinite(value):
            raise ToolError("Result is not a finite number.")
        return value
    raise ToolError("Only numbers, parentheses and arithmetic operators are allowed.")


def calculator(expression: str) -> ToolResult:
    try:
        tree = ast.parse(str(expression), mode="eval")
        value = _safe_number(tree.body)
        text = f"{value:g}" if value != int(value) else str(int(value))
        return ToolResult("calculator", True, text, value)
    except (SyntaxError, ZeroDivisionError, OverflowError, ToolError) as exc:
        return ToolResult("calculator", False, "", error=str(exc))


_FIXED_OFFSETS = {"Asia/Jakarta": 7, "Asia/Makassar": 8, "Asia/Jayapura": 9, "UTC": 0, "Etc/UTC": 0}


def current_datetime(timezone: str | None = None) -> ToolResult:
    try:
        if timezone:
            from zoneinfo import ZoneInfo

            try:
                moment = datetime.now(ZoneInfo(timezone))
            except Exception:
                # Windows Python has no IANA database unless "tzdata" is installed; keep the
                # common Indonesian zones usable with their fixed offsets.
                offset = _FIXED_OFFSETS.get(timezone)
                if offset is None:
                    raise
                from datetime import timedelta, timezone as dt_timezone

                moment = datetime.now(dt_timezone(timedelta(hours=offset), timezone))
        else:
            moment = datetime.now().astimezone()
    except Exception as exc:  # noqa: BLE001
        return ToolResult("current_datetime", False, "", error=f"Unknown timezone: {exc}")
    text = moment.isoformat(timespec="seconds")
    return ToolResult("current_datetime", True, text, {"iso": text, "timezone": str(moment.tzinfo)})


def execute_tool(name: str, arguments: Mapping[str, Any] | None = None, *, rag: Any = None, web: Any = None,
                 collection: str | None = None) -> ToolResult:
    arguments = dict(arguments or {})
    if name == "calculator":
        return calculator(str(arguments.get("expression") or ""))
    if name == "current_datetime":
        return current_datetime(arguments.get("timezone") or None)
    if name == "knowledge_search":
        if rag is None:
            return ToolResult(name, False, "", error="No knowledge base is available.")
        query = str(arguments.get("query") or "").strip()
        target = str(arguments.get("collection") or collection or "").strip()
        if not query or not target:
            return ToolResult(name, False, "", error="query and collection are required.")
        try:
            result = rag.query(target, query, top_k=4)
        except Exception as exc:  # noqa: BLE001
            return ToolResult(name, False, "", error=str(exc))
        hits = result.get("hits") or []
        summary = "\n".join(f"[{hit.get('source')}] {' '.join(str(hit.get('text', '')).split())[:400]}"
                            for hit in hits[:4]) or "No matches."
        return ToolResult(name, True, summary, result)
    if name == "web_search":
        if web is None:
            return ToolResult(name, False, "", error="Web search is not configured.")
        query = str(arguments.get("query") or "").strip()
        if not query:
            return ToolResult(name, False, "", error="query is required.")
        try:
            result = web.search(query)
        except Exception as exc:  # noqa: BLE001
            return ToolResult(name, False, "", error=str(exc))
        summary = "\n".join(f"{item.get('title')}: {item.get('url')}" for item in (result.get("results") or [])[:4])
        return ToolResult(name, True, summary or "No results.", result)
    return ToolResult(name, False, "", error=f"Unknown tool: {name}")


def tool_instructions(enabled: list[str] | tuple[str, ...] | None = None) -> str:
    names = list(enabled) if enabled else [name for name in TOOLS]
    specs = [TOOLS[name].to_dict() for name in names if name in TOOLS]
    return ("You can call tools. To use one, reply with exactly one JSON object inside "
            "<tool_call></tool_call>, for example:\n"
            '<tool_call>{"name":"calculator","arguments":{"expression":"1234*5678"}}</tool_call>\n'
            "Available tools:\n" + json.dumps(specs, indent=2) +
            "\nAfter a tool call, wait for the tool result before answering normally.")


def _coerce_call(raw: Any) -> ToolCall | None:
    if isinstance(raw, list):
        raw = raw[0] if raw else None
    if not isinstance(raw, Mapping):
        return None
    name = raw.get("name") or raw.get("tool") or raw.get("tool_name")
    arguments = raw.get("arguments") or raw.get("parameters") or raw.get("input") or {}
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except ValueError:
            arguments = {}
    if not isinstance(name, str) or name not in TOOLS or not isinstance(arguments, Mapping):
        return None
    return ToolCall(name, dict(arguments))


def parse_tool_call(text: str) -> ToolCall | None:
    """Detect the common tool-call formats in a model response; return the first valid call."""
    if not text:
        return None
    for pattern in (TOOL_CALL_RE, FENCE_RE, MISTRAL_RE):
        for match in pattern.finditer(text):
            try:
                raw = json.loads(match.group(1))
            except ValueError:
                continue
            call = _coerce_call(raw)
            if call:
                return call
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            call = _coerce_call(json.loads(stripped))
        except ValueError:
            call = None
        if call:
            return call
    # Embedded JSON object surrounded by prose (the model may add a sentence first).
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            raw, _end = decoder.raw_decode(text[match.start():])
        except ValueError:
            continue
        call = _coerce_call(raw)
        if call:
            return call
    return None


def validate_call(call: ToolCall) -> None:
    spec = TOOLS.get(call.name)
    if spec is None:
        raise ToolError(f"Unknown tool: {call.name}")
    required = spec.parameters.get("required") or []
    missing = [key for key in required if call.arguments.get(key) in (None, "")]
    if missing:
        raise ToolError(f"Missing required argument(s): {', '.join(missing)}")