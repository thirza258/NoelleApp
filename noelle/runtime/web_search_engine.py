"""Agentic web grounding through a SearXNG instance.

Modelled on HuggingChat (chat-ui) web search: the chat model proposes a few
search queries, SearXNG's JSON API returns links, the pages are fetched in
parallel and reduced to readable text, cut into ~1000-character passages,
ranked against the question (embedding cosine similarity when an embedder is
given, BM25 otherwise) and the best passages are packed into a numbered
context block that the chat model cites as ``[n]``.

Security:

* only http(s) URLs returned by the search provider are fetched;
* page fetches refuse loopback, private, link-local, shared (CGNAT), multicast
  and other non-public addresses: hostnames are resolved, every redirect hop is
  checked, and the connected peer address is re-checked to blunt DNS rebinding
  (that last check is skipped when an HTTP proxy is configured);
* the configured SearXNG URL is trusted for the search call only (it is
  usually a local instance);
* no cookies are stored or sent, bodies are capped at ~2 MB, non-HTML pages
  are skipped. Page text and queries are never logged.
"""

from __future__ import annotations

import html
import ipaddress
import logging
import math
import os
import re
import socket
import threading
import time
from collections import Counter
from collections.abc import Callable, Sequence
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from html.parser import HTMLParser
from http.cookiejar import CookieJar, DefaultCookiePolicy
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx

from noelle.runtime.errors import BadRequest, Cancelled

log = logging.getLogger("noelle.web")

USER_AGENT = "Noelle/1.0 (+local)"
MAX_PAGE_BYTES = 2_000_000
MAX_REDIRECTS = 5
PASSAGE_CHARS = 1000
MAX_PASSAGES_PER_PAGE = 40
MAX_QUERY_CHARS = 200
MAX_SNIPPET_CHARS = 600
FETCH_WORKERS = 4
EMBED_BATCH = 32
FIRST_PASS_PER_SOURCE = 3
MIN_MAIN_CHARS = 200
MIN_TRUNCATED_CHARS = 120
HTML_TYPES = frozenset({"text/html", "application/xhtml+xml"})
ACCEPT_HTML = "text/html,application/xhtml+xml;q=0.9,*/*;q=0.1"
SETUP_HINT = ("Run SearXNG locally (for example: docker run -d -p 8888:8080 searxng/searxng), allow JSON output in "
              "its settings.yml (search: formats: [html, json]) and set its URL in Settings > Web search.")
JSON_HINT = ("Enable SearXNG's JSON API: add 'json' to search.formats in its settings.yml "
             "(formats: [html, json]) and restart SearXNG.")

_PROXY_VARS = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")
_LOCAL_SUFFIXES = (".localhost", ".local", ".internal", ".home.arpa", ".lan")
_TERM_RE = re.compile(r"\w+")
_TAG_RE = re.compile(r"<[^>]+>")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_META_CHARSET_RE = re.compile(rb"""<meta[^>]+charset\s*=\s*["']?\s*([A-Za-z0-9_.:-]+)""", re.IGNORECASE)
_STOPWORDS = frozenset(
    "a an and are as at be been but by can could did do does for from had has have how i if in into is it its "
    "me my of on or our so than that the their them then there these they this to us was we were what when "
    "where which who whom why will with would you your".split())


# ============================================================================
# HTML to text (stdlib html.parser)
# ============================================================================

_SKIP_TAGS = frozenset({"script", "style", "nav", "header", "footer", "aside", "form", "noscript", "svg", "head",
                        "template", "iframe", "object", "canvas", "button", "select", "textarea", "dialog"})
_SKIP_ROLES = frozenset({"navigation", "banner", "contentinfo", "complementary", "search", "menu", "menubar",
                         "dialog", "alertdialog"})
_MAIN_TAGS = frozenset({"main", "article"})
_VOID_TAGS = frozenset({"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param",
                        "source", "track", "wbr"})
_BLOCK_TAGS = frozenset({"address", "article", "aside", "blockquote", "br", "caption", "dd", "details", "div",
                         "dl", "dt", "figcaption", "figure", "footer", "h1", "h2", "h3", "h4", "h5", "h6", "header",
                         "hr", "li", "main", "nav", "ol", "p", "pre", "section", "summary", "table", "tbody",
                         "tfoot", "thead", "tr", "ul"})
_CELL_TAGS = frozenset({"td", "th"})


class _TextExtractor(HTMLParser):
    """Collects visible text line by line, tracking whether each line sits in <main>/<article>."""

    def __init__(self, *, skip_forms: bool = True) -> None:
        super().__init__(convert_charrefs=True)
        self._skip_forms = skip_forms
        self._stack: list[tuple[str, bool, bool]] = []  # (tag, skips content, is main)
        self._skip = 0
        self._main = 0
        self._buffer: list[str] = []
        self._buffer_main = False
        self.lines: list[str] = []
        self.main_lines: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "body" or tag in _BLOCK_TAGS:
            self._close("head")  # </head> is optional; body content must not inherit its skip flag
        if tag in _BLOCK_TAGS:
            self._flush()
        elif tag in _CELL_TAGS and "".join(self._buffer).strip():
            self._buffer.append(" | ")
        if tag in _VOID_TAGS:
            return
        attributes = {key: (value or "") for key, value in attrs}
        is_main = tag in _MAIN_TAGS or attributes.get("role", "").lower() == "main"
        skip = self._is_boilerplate(tag, attributes)
        if tag == "header" and self._main:  # an article's own header holds its title
            skip = False
        self._stack.append((tag, skip, is_main))
        self._skip += int(skip)
        self._main += int(is_main)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _BLOCK_TAGS:
            self._flush()

    def handle_endtag(self, tag: str) -> None:
        if tag in _BLOCK_TAGS:
            self._flush()
        self._close(tag)

    def _close(self, tag: str) -> None:
        """Pop the innermost open ``tag`` and everything left open inside it; ignore stray end tags."""
        for depth in range(len(self._stack) - 1, -1, -1):
            if self._stack[depth][0] == tag:
                break
        else:
            return
        while len(self._stack) > depth:
            _, skip, is_main = self._stack.pop()
            self._skip -= int(skip)
            self._main -= int(is_main)

    def handle_data(self, data: str) -> None:
        if self._skip:
            return
        if not self._buffer:
            self._buffer_main = self._main > 0
        self._buffer.append(data)

    def text(self) -> str:
        self._flush()
        main = "\n".join(self.main_lines)
        return main if len(main) >= MIN_MAIN_CHARS else "\n".join(self.lines)

    def _is_boilerplate(self, tag: str, attributes: dict[str, str]) -> bool:
        if tag == "form":
            return self._skip_forms
        if tag in _SKIP_TAGS or attributes.get("role", "").lower() in _SKIP_ROLES:
            return True
        return "hidden" in attributes or attributes.get("aria-hidden", "").lower() == "true"

    def _flush(self) -> None:
        if not self._buffer:
            return
        line = " ".join("".join(self._buffer).split())
        self._buffer = []
        if not line:
            return
        if not self.lines or self.lines[-1] != line:
            self.lines.append(line)
        if self._buffer_main and (not self.main_lines or self.main_lines[-1] != line):
            self.main_lines.append(line)


def html_to_text(markup: str) -> str:
    """Readable text of an HTML page: boilerplate dropped, <main>/<article> preferred, one block per line."""
    text = _extract(markup, skip_forms=True)
    if len(text) >= MIN_MAIN_CHARS:
        return text
    unwrapped = _extract(markup, skip_forms=False)  # pages wrapped in one big <form> (ASP.NET WebForms)
    return unwrapped if len(unwrapped) > len(text) else text


def _extract(markup: str, *, skip_forms: bool) -> str:
    parser = _TextExtractor(skip_forms=skip_forms)
    try:
        parser.feed(markup)
        parser.close()
    except Exception as exc:  # html.parser is lenient; keep whatever was collected
        log.debug("HTML parse error: %s", type(exc).__name__)
    return parser.text()


# ============================================================================
# passages and ranking
# ============================================================================


def split_passages(text: str, size: int = PASSAGE_CHARS) -> list[str]:
    """Cut text into passages of at most ``size`` characters, keeping lines and sentences whole when possible."""
    passages: list[str] = []
    lines: list[str] = []
    length = 0
    for raw in text.split("\n"):
        line = raw.strip()
        if not line:
            continue
        for piece in _split_long(line, size):
            if lines and length + 1 + len(piece) > size:
                passages.append("\n".join(lines))
                lines, length = [], 0
            length += len(piece) + (1 if lines else 0)
            lines.append(piece)
    if lines:
        passages.append("\n".join(lines))
    return passages


def _split_long(line: str, size: int) -> list[str]:
    if len(line) <= size:
        return [line]
    pieces: list[str] = []
    current = ""
    for sentence in _SENTENCE_SPLIT_RE.split(line):
        while len(sentence) > size:  # hard-wrap a run-on sentence at a space
            cut = sentence.rfind(" ", 0, size)
            cut = cut if cut > size // 2 else size
            if current:
                pieces.append(current)
                current = ""
            pieces.append(sentence[:cut].rstrip())
            sentence = sentence[cut:].lstrip()
        if current and len(current) + 1 + len(sentence) > size:
            pieces.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        pieces.append(current)
    return pieces


def _terms(text: str) -> list[str]:
    return [t for t in _TERM_RE.findall(text.lower()) if t not in _STOPWORDS]


def bm25_scores(query: str, documents: Sequence[str], *, k1: float = 1.5, b: float = 0.75) -> list[float]:
    """Okapi BM25 score of each document for ``query``."""
    wanted = set(_terms(query)) or set(_TERM_RE.findall(query.lower()))
    if not wanted or not documents:
        return [0.0] * len(documents)
    tokenized = [_TERM_RE.findall(doc.lower()) for doc in documents]
    count = len(tokenized)
    average = (sum(len(tokens) for tokens in tokenized) / count) or 1.0
    frequency = Counter(term for tokens in tokenized for term in set(tokens) if term in wanted)
    scores: list[float] = []
    for tokens in tokenized:
        counts = Counter(term for term in tokens if term in wanted)
        score = 0.0
        for term, tf in counts.items():
            idf = math.log(1.0 + (count - frequency[term] + 0.5) / (frequency[term] + 0.5))
            score += idf * tf * (k1 + 1.0) / (tf + k1 * (1.0 - b + b * len(tokens) / average))
        scores.append(score)
    return scores


def _embedding_scores(question: str, passages: Sequence[str], embedder: Any,
                      stop_event: threading.Event | None) -> list[float]:
    import numpy as np

    query = np.asarray(embedder.encode([question], normalize=True), dtype=np.float32).reshape(-1)
    query = query / max(float(np.linalg.norm(query)), 1e-12)
    scores: list[float] = []
    for begin in range(0, len(passages), EMBED_BATCH):
        _check_stop(stop_event)
        matrix = np.asarray(embedder.encode(list(passages[begin:begin + EMBED_BATCH]), normalize=True),
                            dtype=np.float32)
        matrix = matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
        scores.extend(float(value) for value in matrix @ query)
    return scores


@dataclass
class _Candidate:
    page: int  # index into the page list
    position: int  # passage order inside the page
    text: str
    score: float = 0.0


def _render(chosen: Sequence[_Candidate], pages: Sequence[dict[str, str]], start_index: int) -> tuple[str, list[int]]:
    """Context text grouped by source, sources numbered by first appearance from ``start_index``."""
    order: list[int] = []
    groups: dict[int, list[_Candidate]] = {}
    for candidate in chosen:
        if candidate.page not in groups:
            order.append(candidate.page)
            groups[candidate.page] = []
        groups[candidate.page].append(candidate)
    blocks: list[str] = []
    for number, page_index in enumerate(order, start=start_index):
        page = pages[page_index]
        title, url = page["title"], page["url"]
        header = f"[{number}] {title} ({url})" if title and title != url else f"[{number}] {url}"
        body = "\n\n".join(c.text for c in sorted(groups[page_index], key=lambda c: c.position))
        blocks.append(f"{header}\n{body}")
    return "\n\n".join(blocks), order


def _select(ranked: Sequence[_Candidate], pages: Sequence[dict[str, str]], budget: int,
            start_index: int) -> list[_Candidate]:
    """Greedy best-first packing under ``budget`` characters; first pass caps passages per source."""
    chosen: list[_Candidate] = []
    taken: set[tuple[int, int]] = set()
    per_page: Counter[int] = Counter()
    if budget <= 0:
        return chosen
    for first_pass in (True, False):
        for candidate in ranked:
            key = (candidate.page, candidate.position)
            if key in taken or (first_pass and per_page[candidate.page] >= FIRST_PASS_PER_SOURCE):
                continue
            if len(_render([*chosen, candidate], pages, start_index)[0]) <= budget:
                chosen.append(candidate)
            elif not chosen:  # make the single best passage fit a small budget
                header = len(_render([_Candidate(candidate.page, 0, "")], pages, start_index)[0])
                room = budget - header
                if room < MIN_TRUNCATED_CHARS:
                    continue
                chosen.append(_Candidate(candidate.page, candidate.position, _truncate(candidate.text, room),
                                         candidate.score))
            else:
                continue
            taken.add(key)
            per_page[candidate.page] += 1
    return chosen


# ============================================================================
# network guard
# ============================================================================


def _is_public_ip(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
        address = address.ipv4_mapped
    return address.is_global and not address.is_multicast


def _parse_ip(text: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address:
    return ipaddress.ip_address(text.split("%", 1)[0].strip("[]"))


def _is_public_host(host: str, port: int) -> bool:
    """Whether every address ``host`` resolves to is publicly routable."""
    name = host.lower().rstrip(".")
    if not name or name == "localhost" or name.endswith(_LOCAL_SUFFIXES):
        return False
    try:
        return _is_public_ip(_parse_ip(name))
    except ValueError:
        pass
    try:
        infos = socket.getaddrinfo(name, port, type=socket.SOCK_STREAM)
    except (OSError, UnicodeError) as exc:
        raise BadRequest(f"Could not resolve {host}.") from exc
    addresses = {str(info[4][0]) for info in infos}
    try:
        return bool(addresses) and all(_is_public_ip(_parse_ip(address)) for address in addresses)
    except ValueError:
        return False


# ============================================================================
# engine
# ============================================================================


class WebSearchEngine:
    """SearXNG search, safe page fetching and passage ranking for chat grounding."""

    def __init__(self, searxng_url: str | None, *, timeout: float = 10, results_per_query: int = 4,
                 context_chars: int = 8000, allow_private_hosts: bool = False, max_pages: int = 8) -> None:
        self.searxng_url = (searxng_url or "").strip()
        self.timeout = max(float(timeout or 10), 0.5)
        self.results_per_query = max(1, int(results_per_query or 4))
        self.context_chars = max(0, int(context_chars or 0))
        self.allow_private_hosts = allow_private_hosts
        self.max_pages = max(1, int(max_pages or 1))
        self._via_proxy = any(os.environ.get(name) for name in _PROXY_VARS)

    # ----------------------------------------------------------------- search
    def search(self, query: str) -> list[dict[str, str]]:
        """``[{"title", "url", "snippet"}]`` from SearXNG's JSON API (http(s) links only, deduplicated)."""
        query = " ".join(str(query or "").split())[:MAX_QUERY_CHARS]
        if not query:
            return []
        endpoint = self._search_endpoint()
        try:
            with self._client(follow_redirects=True) as client:
                response = client.get(endpoint, params={"q": query, "format": "json"},
                                      headers={"Accept": "application/json"})
        except httpx.TimeoutException as exc:
            raise BadRequest(f"SearXNG at {endpoint} did not answer within {self.timeout:g} s.",
                             hint=SETUP_HINT) from exc
        except httpx.HTTPError as exc:
            raise BadRequest(f"Could not reach SearXNG at {endpoint} ({type(exc).__name__}).", hint=SETUP_HINT) from exc
        if response.status_code == 403:
            raise BadRequest("SearXNG refused the request (HTTP 403); its JSON output is probably disabled.",
                             hint=JSON_HINT)
        if response.status_code == 429:
            raise BadRequest("SearXNG rate-limited the request (HTTP 429).",
                             hint="For a private instance set server.limiter: false in SearXNG's settings.yml.")
        if response.status_code >= 400:
            raise BadRequest(f"SearXNG returned HTTP {response.status_code}.", hint=SETUP_HINT)
        try:
            data = response.json()
        except ValueError as exc:
            raise BadRequest("SearXNG did not return JSON.", hint=JSON_HINT) from exc
        results = data.get("results") if isinstance(data, dict) else None
        if not isinstance(results, list):
            raise BadRequest("SearXNG returned an unexpected response (no results list).", hint=JSON_HINT)
        found: list[dict[str, str]] = []
        seen: set[str] = set()
        for item in results:
            if not isinstance(item, dict):
                continue
            url = str(item.get("url") or "").strip()
            key = _url_key(url)
            if key is None or key in seen:
                continue
            seen.add(key)
            found.append({"title": _clean(item.get("title"), 300) or url, "url": url,
                          "snippet": _clean(item.get("content"), MAX_SNIPPET_CHARS)})
            if len(found) >= self.results_per_query:
                break
        return found

    # ------------------------------------------------------------------ fetch
    def fetch_text(self, url: str) -> str:
        """Readable text of an HTML page; raises :class:`BadRequest` for refused, failed or non-HTML fetches."""
        current = str(url or "").strip()
        with self._client(follow_redirects=False) as client:
            for _hop in range(MAX_REDIRECTS + 1):
                self._guard_url(current)
                location, body, charset = self._get_once(client, current)
                if location is None:
                    return html_to_text(_decode(body, charset))
                current = location
        raise BadRequest("The page redirected too many times.")

    # ----------------------------------------------------------------- ground
    def ground(self, question: str, *, generate_queries: Callable[[str], list[str]] | None = None,
               embedder: Any = None, max_queries: int = 3, on_status: Callable[[str], None] | None = None,
               stop_event: threading.Event | None = None, start_index: int = 1) -> dict[str, Any]:
        """Search, read and rank the web for ``question``.

        Returns ``{"queries", "sources", "passages", "context"}``. Sources are
        exactly the pages quoted in ``context``, numbered contiguously from
        ``start_index`` in the order they appear; each passage carries its
        source's number as ``index``.
        """
        status = on_status or _ignore
        question = " ".join(str(question or "").split())
        outcome: dict[str, Any] = {"queries": [], "sources": [], "passages": [], "context": ""}
        if not question:
            return outcome
        _check_stop(stop_event)
        queries = _plan_queries(question, generate_queries, max_queries)
        outcome["queries"] = queries
        _check_stop(stop_event)
        pages = self._search_all(queries, status, stop_event)
        if not pages:
            status("No web results found")
            return outcome
        status(f"Reading {len(pages)} page{'s' if len(pages) != 1 else ''}")
        texts = self._fetch_all([page["url"] for page in pages], stop_event)
        candidates: list[_Candidate] = []
        for index, page in enumerate(pages):
            text = texts.get(page["url"], "")
            pieces = split_passages(text)[:MAX_PASSAGES_PER_PAGE] if text else []
            if not pieces and page["snippet"]:  # unreadable page: fall back to the search snippet
                pieces = [page["snippet"]]
            candidates.extend(_Candidate(index, position, piece) for position, piece in enumerate(pieces))
        if not candidates:
            return outcome
        _check_stop(stop_event)
        status("Ranking passages")
        for candidate, score in zip(candidates, self._score(question, candidates, embedder, stop_event)):
            candidate.score = float(score)
        if any(candidate.score > 0 for candidate in candidates):  # drop unrelated pages, keep search order otherwise
            candidates = [candidate for candidate in candidates if candidate.score > 0]
        ranked = sorted(candidates, key=lambda c: (-c.score, c.page, c.position))
        chosen = _select(ranked, pages, self.context_chars, start_index)
        context, order = _render(chosen, pages, start_index)
        numbers = {page_index: start_index + offset for offset, page_index in enumerate(order)}
        outcome["sources"] = [{"index": numbers[i], "title": pages[i]["title"], "url": pages[i]["url"],
                               "snippet": pages[i]["snippet"]} for i in order]
        outcome["passages"] = [
            {"index": numbers[c.page], "url": pages[c.page]["url"], "title": pages[c.page]["title"],
             "text": c.text, "score": round(c.score, 4)}
            for c in sorted(chosen, key=lambda c: (-c.score, c.page, c.position))]
        outcome["context"] = context
        return outcome

    # --------------------------------------------------------------- helpers
    def _search_all(self, queries: list[str], status: Callable[[str], None],
                    stop_event: threading.Event | None) -> list[dict[str, str]]:
        """Run every query; results are interleaved by rank across queries, deduplicated and capped."""
        per_query: list[list[dict[str, str]]] = []
        errors: list[BadRequest] = []
        for query in queries:
            _check_stop(stop_event)
            status(f"Searching: {query}")
            try:
                per_query.append(self.search(query))
            except BadRequest as exc:
                errors.append(exc)
                log.info("web search failed: %s", exc.message)
        if errors and not per_query:
            raise errors[0]
        pages: list[dict[str, str]] = []
        seen: set[str] = set()
        for rank in range(max((len(results) for results in per_query), default=0)):
            for results in per_query:
                if rank < len(results):
                    key = _url_key(results[rank]["url"])
                    if key is not None and key not in seen:
                        seen.add(key)
                        pages.append(dict(results[rank]))
        return pages[:self.max_pages]

    def _fetch_all(self, urls: list[str], stop_event: threading.Event | None) -> dict[str, str]:
        texts: dict[str, str] = {}
        if not urls:
            return texts
        pool = ThreadPoolExecutor(max_workers=min(FETCH_WORKERS, len(urls)), thread_name_prefix="noelle-web")
        try:
            futures = {pool.submit(self._fetch_quietly, url): url for url in urls}
            pending = set(futures)
            while pending:
                _check_stop(stop_event)
                done, pending = wait(pending, timeout=0.2, return_when=FIRST_COMPLETED)
                for future in done:
                    text = future.result()
                    if text:
                        texts[futures[future]] = text
        finally:
            pool.shutdown(wait=False, cancel_futures=True)
        return texts

    def _fetch_quietly(self, url: str) -> str:
        try:
            return self.fetch_text(url)
        except Exception as exc:  # one bad page must not sink the search
            log.debug("skipped %s: %s", url, getattr(exc, "message", type(exc).__name__))
            return ""

    def _score(self, question: str, candidates: Sequence[_Candidate], embedder: Any,
               stop_event: threading.Event | None) -> list[float]:
        texts = [candidate.text for candidate in candidates]
        if embedder is not None:
            try:
                return _embedding_scores(question, texts, embedder, stop_event)
            except Cancelled:
                raise
            except Exception as exc:
                log.warning("embedding ranking failed (%s); using BM25", type(exc).__name__)
        return bm25_scores(question, texts)

    def _search_endpoint(self) -> str:
        base = self.searxng_url
        if not base:
            raise BadRequest("Web search needs a SearXNG instance, but no SearXNG URL is configured.",
                             hint=SETUP_HINT)
        if "://" not in base:
            base = f"http://{base}"
        try:
            parts = urlsplit(base)
        except ValueError as exc:
            raise BadRequest(f"'{self.searxng_url}' is not a valid SearXNG URL.", hint=SETUP_HINT) from exc
        if parts.scheme.lower() not in ("http", "https") or not parts.netloc:
            raise BadRequest(f"'{self.searxng_url}' is not a valid SearXNG URL.", hint=SETUP_HINT)
        path = parts.path.rstrip("/")
        if path.endswith("/search"):
            path = path[:-len("/search")]
        return urlunsplit((parts.scheme, parts.netloc, f"{path}/search", "", ""))

    def _client(self, *, follow_redirects: bool) -> httpx.Client:
        return httpx.Client(timeout=httpx.Timeout(self.timeout), follow_redirects=follow_redirects,
                            max_redirects=MAX_REDIRECTS, headers={"User-Agent": USER_AGENT},
                            cookies=CookieJar(policy=DefaultCookiePolicy(allowed_domains=[])))

    def _guard_url(self, url: str) -> None:
        try:
            parsed = httpx.URL(url)
        except Exception as exc:
            raise BadRequest("Invalid URL.") from exc
        if parsed.scheme not in ("http", "https"):
            raise BadRequest("Only http(s) pages can be fetched.")
        if not parsed.host:
            raise BadRequest("The URL has no host.")
        if parsed.userinfo:
            raise BadRequest("URLs with embedded credentials are not fetched.")
        if self.allow_private_hosts:
            return
        host = parsed.raw_host.decode("ascii", errors="replace")
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        if not _is_public_host(host, port):
            raise BadRequest(f"Refusing to fetch {parsed.host}: it is not a public internet address.")

    def _guard_peer(self, response: httpx.Response) -> None:
        """Re-check the address actually connected to (DNS rebinding); skipped behind a proxy."""
        if self.allow_private_hosts or self._via_proxy:
            return
        stream = response.extensions.get("network_stream")
        try:
            peer = stream.get_extra_info("server_addr") if stream is not None else None
        except Exception:
            peer = None
        if not peer:
            return
        try:
            public = _is_public_ip(_parse_ip(str(peer[0])))
        except ValueError:
            public = False
        if not public:
            raise BadRequest("Refusing to read a page served from a private network address.")

    def _get_once(self, client: httpx.Client, url: str) -> tuple[str | None, bytes, str | None]:
        """One GET without redirects: ``(redirect location, body, charset)``."""
        try:
            with client.stream("GET", url, headers={"Accept": ACCEPT_HTML}) as response:
                self._guard_peer(response)
                if response.is_redirect:
                    return str(response.url.join(response.headers["location"])), b"", None
                if response.status_code >= 400:
                    raise BadRequest(f"The page returned HTTP {response.status_code}.")
                content_type = response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
                if content_type and content_type not in HTML_TYPES:
                    raise BadRequest(f"Skipped non-HTML content ({content_type}).")
                return None, self._read_body(response), response.charset_encoding
        except (httpx.HTTPError, httpx.InvalidURL) as exc:
            raise BadRequest(f"Could not fetch the page ({type(exc).__name__}).") from exc

    def _read_body(self, response: httpx.Response) -> bytes:
        deadline = time.monotonic() + self.timeout * 2
        parts: list[bytes] = []
        size = 0
        for chunk in response.iter_bytes():
            room = MAX_PAGE_BYTES - size
            parts.append(chunk[:room])
            size += min(len(chunk), room)
            if size >= MAX_PAGE_BYTES or time.monotonic() > deadline:
                break
        return b"".join(parts)


# ============================================================================
# small helpers
# ============================================================================


def _plan_queries(question: str, generate_queries: Callable[[str], list[str]] | None, max_queries: int) -> list[str]:
    """Model-written queries, cleaned and deduplicated; falls back to the question itself."""
    limit = max(1, int(max_queries or 1))
    proposed: list[str] = []
    if generate_queries is not None:
        try:
            result = generate_queries(question)
            proposed = result.splitlines() if isinstance(result, str) else list(result or [])
        except Cancelled:
            raise
        except Exception as exc:
            log.warning("query generation failed (%s); searching for the question", type(exc).__name__)
    queries: list[str] = []
    seen: set[str] = set()
    for item in proposed:
        query = " ".join(str(item).split())[:MAX_QUERY_CHARS].strip()
        if query and query.lower() not in seen:
            seen.add(query.lower())
            queries.append(query)
        if len(queries) >= limit:
            break
    return queries or [_truncate(question, MAX_QUERY_CHARS).rstrip("…")]


def _url_key(url: str) -> str | None:
    """Dedup key for http(s) URLs (fragment dropped, scheme/host lower-cased); ``None`` for anything else."""
    try:
        parts = urlsplit(url)
    except ValueError:
        return None
    if parts.scheme.lower() not in ("http", "https") or not parts.netloc:
        return None
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path or "/", parts.query, ""))


def _clean(value: Any, limit: int) -> str:
    text = " ".join(html.unescape(_TAG_RE.sub(" ", str(value or ""))).split())
    return _truncate(text, limit)


def _decode(body: bytes, charset: str | None) -> str:
    encoding = charset
    if not encoding:
        match = _META_CHARSET_RE.search(body[:4096])
        encoding = match.group(1).decode("ascii", errors="ignore") if match else "utf-8"
    try:
        return body.decode(encoding, errors="replace")
    except LookupError:
        return body.decode("utf-8", errors="replace")


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    cut = text[:max(0, limit - 1)]
    space = cut.rfind(" ")
    if space > limit * 0.6:
        cut = cut[:space]
    return cut.rstrip() + "…"


def _check_stop(stop_event: threading.Event | None) -> None:
    if stop_event is not None and stop_event.is_set():
        raise Cancelled("Web search was cancelled.")


def _ignore(_message: str) -> None:
    return None
