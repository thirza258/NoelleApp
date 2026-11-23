"""WebSearchEngine against a local fake SearXNG + website (no internet access needed)."""

from __future__ import annotations

import json
import re
import socket
import threading
import zlib
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlsplit

import numpy as np
import pytest

from noelle.runtime.errors import BadRequest, Cancelled
from noelle.runtime.web_search_engine import (
    USER_AGENT,
    WebSearchEngine,
    bm25_scores,
    html_to_text,
    split_passages,
)

SOLAR_PAGE = """<!doctype html><html><head><title>Solar panels</title><script>track('visit')</script></head>
<body><nav><a href="/">Home</a> <a href="/shop">Shop</a></nav><header>Energy Weekly</header>
<main><article><h1>How efficient are solar panels?</h1>
<p>Modern photovoltaic solar panels convert between 18 and 23 percent of sunlight into electricity.</p>
<p>Panel   efficiency drops slightly as the cells get hotter,
   so cool climates help.</p>
<form><input name="email"> Subscribe to our newsletter</form>
</article></main><aside>Sponsored: buy a boat</aside><footer>Copyright 2026 Energy Weekly</footer></body></html>"""

GARDEN_PAGE = ("<html><body><main><h1>Gardening</h1><p>Tomatoes need sunlight, water and well drained soil.</p>"
               "<p>Prune the side shoots of tomato plants every week.</p></main></body></html>")

WIND_PAGE = ("<html><body><article><h2>Wind turbines</h2><p>Wind turbines convert the kinetic energy of moving "
             "air into electricity.</p></article></body></html>")

PAGES = {"/solar": SOLAR_PAGE, "/garden": GARDEN_PAGE, "/wind": WIND_PAGE}


class FakeEmbedder:
    """Deterministic hashed bag-of-words vectors (zlib.crc32, never the per-process salted ``hash``)."""

    def __init__(self, dim: int = 1024) -> None:
        self.dim = dim
        self.calls = 0

    def encode(self, texts: list[str], normalize: bool = True, **_kwargs: Any) -> np.ndarray:
        self.calls += 1
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in re.findall(r"\w+", text.lower()):
                out[row, zlib.crc32(word.encode()) % self.dim] += 1.0
        if normalize:
            out /= np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)
        return out


@dataclass
class Site:
    base: str = ""
    hits: Counter[str] = field(default_factory=Counter)
    cookies: list[str | None] = field(default_factory=list)
    agents: list[str | None] = field(default_factory=list)


@pytest.fixture
def site() -> Iterator[Site]:
    """Fake SearXNG (``/search``) plus the pages its results point at, on 127.0.0.1."""
    state = Site()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args: Any) -> None:
            pass

        def _send(self, status: int, body: str | bytes, content_type: str, **headers: str) -> None:
            data = body.encode() if isinstance(body, str) else body
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            for key, value in headers.items():
                self.send_header(key.replace("_", "-"), value)
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:  # noqa: N802 - http.server naming
            parts = urlsplit(self.path)
            state.hits[parts.path] += 1
            state.cookies.append(self.headers.get("Cookie"))
            state.agents.append(self.headers.get("User-Agent"))
            base = state.base
            if parts.path == "/search":
                params = parse_qs(parts.query)
                if params.get("format") != ["json"]:
                    return self._send(400, "html only", "text/html")
                results = [
                    {"title": "Solar panel efficiency", "url": f"{base}/solar",
                     "content": "How efficient are <b>solar</b> panels?"},
                    {"title": "Solar again", "url": f"{base}/solar#top", "content": "duplicate"},
                    {"title": "FTP mirror", "url": "ftp://example.com/x", "content": "ignored"},
                    {"title": "Gardening tips", "url": f"{base}/garden", "content": "Tomatoes and sunlight"},
                    {"title": "Wind power", "url": f"{base}/wind", "content": "Wind turbines and electricity"},
                    {"title": "Solar datasheet", "url": f"{base}/doc.pdf",
                     "content": "Datasheet: solar panels efficiency 22 percent"},
                ]
                return self._send(200, json.dumps({"query": params.get("q"), "results": results}),
                                  "application/json", Set_Cookie="searxng_pref=1")
            if parts.path == "/forbidden/search":
                return self._send(403, "Forbidden", "text/html")
            if parts.path == "/redirect":
                return self._send(302, "", "text/html", Location="/solar", Set_Cookie="sid=secret")
            if parts.path == "/loop":
                return self._send(302, "", "text/html", Location="/loop")
            if parts.path == "/doc.pdf":
                return self._send(200, b"%PDF-1.4 binary", "application/pdf")
            if parts.path in PAGES:
                return self._send(200, PAGES[parts.path], "text/html; charset=utf-8")
            return self._send(404, "missing", "text/html")

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    state.base = f"http://127.0.0.1:{server.server_address[1]}"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


# ============================================================================
# text extraction, passages, BM25
# ============================================================================


def test_html_to_text_drops_boilerplate() -> None:
    lines = html_to_text(SOLAR_PAGE).split("\n")
    assert lines == [
        "How efficient are solar panels?",
        "Modern photovoltaic solar panels convert between 18 and 23 percent of sunlight into electricity.",
        "Panel efficiency drops slightly as the cells get hotter, so cool climates help.",
    ]


def test_html_to_text_prefers_main_and_survives_sloppy_markup() -> None:
    markup = ("<html><head><title>x</title><body><div>Trending: celebrity gossip</div><main>"
              + "".join(f"<p>Paragraph {i} explains how river deltas form.</p>" for i in range(8))
              + "</main><div role='contentinfo'>Legal</div></body></html>")
    text = html_to_text(markup)  # no </head>, so <body> must close it implicitly
    assert "celebrity" not in text and "Legal" not in text
    assert text.splitlines()[0] == "Paragraph 0 explains how river deltas form."
    assert html_to_text("<h2>Title</h2><p>Some   spaced\n text <i>here</i></p><ul><li>one<li>two</ul>") == \
        "Title\nSome spaced text here\none\ntwo"
    wrapped = "<form id='aspnetForm'><h1>Report</h1><p>" + "Revenue grew strongly. " * 12 + "</p></form>"
    assert html_to_text(wrapped).startswith("Report\nRevenue grew strongly.")  # whole-page <form> wrapper


def test_split_passages_and_bm25() -> None:
    text = "\n".join(f"Line {i} " + "word " * 30 for i in range(20))
    passages = split_passages(text, 300)
    assert len(passages) > 3 and all(len(p) <= 300 for p in passages)
    assert all(len(p) <= 250 for p in split_passages("Sentence one is here. " * 100, 250))
    scores = bm25_scores("solar panels", ["solar panels are great", "wind turbines", "solar"])
    assert scores[0] > scores[2] > scores[1] == 0.0


# ============================================================================
# search
# ============================================================================


def test_search_parses_dedupes_and_caps(site: Site) -> None:
    engine = WebSearchEngine(site.base, results_per_query=4)  # local SearXNG is allowed for the search call
    results = engine.search("solar panels")
    assert [r["url"] for r in results] == [f"{site.base}/solar", f"{site.base}/garden", f"{site.base}/wind",
                                           f"{site.base}/doc.pdf"]
    assert results[0] == {"title": "Solar panel efficiency", "url": f"{site.base}/solar",
                          "snippet": "How efficient are solar panels?"}
    assert len(WebSearchEngine(site.base + "/search/", results_per_query=2).search("solar")) == 2
    assert engine.search("   ") == []
    assert set(site.agents) == {USER_AGENT} and site.cookies == [None, None]


def test_search_errors_carry_hints(site: Site) -> None:
    with pytest.raises(BadRequest) as forbidden:
        WebSearchEngine(f"{site.base}/forbidden").search("solar")
    assert "403" in forbidden.value.message and "json" in (forbidden.value.hint or "").lower()
    with pytest.raises(BadRequest) as unconfigured:
        WebSearchEngine(None).search("solar")
    assert unconfigured.value.hint
    with pytest.raises(BadRequest) as down:
        WebSearchEngine(f"http://127.0.0.1:{free_port()}", timeout=2).search("solar")
    assert "reach" in down.value.message and down.value.hint


# ============================================================================
# fetching + SSRF guard
# ============================================================================


def test_private_targets_are_refused_by_default(site: Site) -> None:
    engine = WebSearchEngine(site.base)
    for url in (f"{site.base}/solar", "http://localhost:9/", "http://169.254.169.254/latest/meta-data",
                "http://[::1]/", "http://10.0.0.1/", "http://printer.local/", "file:///etc/passwd",
                "ftp://example.com/x", "http://user:pw@example.com/"):
        with pytest.raises(BadRequest):
            engine.fetch_text(url)
    assert site.hits["/solar"] == 0


def test_hostnames_are_resolved_before_fetching(monkeypatch: pytest.MonkeyPatch) -> None:
    def private_dns(host: str, port: Any, *args: Any, **kwargs: Any) -> list[Any]:
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("192.168.1.10", int(port or 80)))]

    monkeypatch.setattr(socket, "getaddrinfo", private_dns)
    with pytest.raises(BadRequest, match="not a public"):
        WebSearchEngine("http://127.0.0.1:1").fetch_text("http://intranet.example.com/")


def test_dns_rebinding_is_caught_after_connect(site: Site, monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.delenv(name, raising=False)
    real_getaddrinfo = socket.getaddrinfo
    answers = iter(["93.184.216.34"])  # first lookup (the guard) sees a public address, then loopback

    def rebinding_dns(host: Any, port: Any, *args: Any, **kwargs: Any) -> list[Any]:
        if host == "rebind.test":
            address = next(answers, "127.0.0.1")
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, int(port)))]
        return real_getaddrinfo(host, port, *args, **kwargs)

    monkeypatch.setattr(socket, "getaddrinfo", rebinding_dns)
    port = urlsplit(site.base).port
    with pytest.raises(BadRequest, match="private network"):
        WebSearchEngine(site.base).fetch_text(f"http://rebind.test:{port}/solar")


def test_fetch_follows_redirects_without_cookies_and_skips_non_html(site: Site) -> None:
    engine = WebSearchEngine(site.base, allow_private_hosts=True)
    assert "photovoltaic" in engine.fetch_text(f"{site.base}/redirect")
    with pytest.raises(BadRequest, match="non-HTML"):
        engine.fetch_text(f"{site.base}/doc.pdf")
    with pytest.raises(BadRequest, match="too many"):
        engine.fetch_text(f"{site.base}/loop")
    with pytest.raises(BadRequest, match="404"):
        engine.fetch_text(f"{site.base}/nowhere")
    assert site.hits["/loop"] == 6  # the first request plus five redirects
    assert all(cookie is None for cookie in site.cookies)
    assert set(site.agents) == {USER_AGENT}


# ============================================================================
# ground()
# ============================================================================


def test_ground_lexical_numbering_and_budget(site: Site) -> None:
    engine = WebSearchEngine(site.base, allow_private_hosts=True, context_chars=450)
    statuses: list[str] = []
    out = engine.ground(
        "How efficient are solar panels?",
        generate_queries=lambda q: ["solar panel efficiency", "Solar  panel efficiency", "", "solar panels heat",
                                    "never used"],
        embedder=None, max_queries=2, on_status=statuses.append, start_index=3)

    assert out["queries"] == ["solar panel efficiency", "solar panels heat"]
    assert statuses == ["Searching: solar panel efficiency", "Searching: solar panels heat", "Reading 4 pages",
                        "Ranking passages"]
    context = out["context"]
    assert 0 < len(context) <= 450
    indexes = [source["index"] for source in out["sources"]]
    assert indexes == list(range(3, 3 + len(indexes))) and len(indexes) >= 2
    for source in out["sources"]:
        assert set(source) == {"index", "title", "url", "snippet"}
        assert f"[{source['index']}] {source['title']} ({source['url']})" in context
    assert out["sources"][0]["url"] == f"{site.base}/solar"
    best = out["passages"][0]
    assert set(best) == {"index", "url", "title", "text", "score"}
    assert best["index"] == 3 and "photovoltaic" in best["text"]
    assert {p["index"] for p in out["passages"]} == set(indexes)
    pdf = next(s for s in out["sources"] if s["url"].endswith("/doc.pdf"))  # unreadable page: snippet used
    assert "Datasheet: solar panels efficiency 22 percent" in context and pdf["index"] in indexes
    assert "Sponsored" not in context and "Copyright" not in context


def test_ground_ranks_with_embedder(site: Site) -> None:
    embedder = FakeEmbedder()
    out = WebSearchEngine(site.base, allow_private_hosts=True).ground(
        "wind turbines kinetic energy", generate_queries=None, embedder=embedder)
    assert out["queries"] == ["wind turbines kinetic energy"]
    assert embedder.calls >= 2
    assert out["sources"][0] == {"index": 1, "title": "Wind power", "url": f"{site.base}/wind",
                                 "snippet": "Wind turbines and electricity"}
    assert out["passages"][0]["index"] == 1 and "kinetic" in out["passages"][0]["text"]
    scores = [p["score"] for p in out["passages"]]
    assert scores == sorted(scores, reverse=True)
    assert out["context"].startswith(f"[1] Wind power ({site.base}/wind)\n")


def test_ground_never_fetches_private_pages_by_default(site: Site) -> None:
    out = WebSearchEngine(site.base).ground("solar panels efficiency")
    assert site.hits["/search"] == 1
    assert all(site.hits[path] == 0 for path in ("/solar", "/garden", "/wind", "/doc.pdf"))
    snippets = {source["index"]: source["snippet"] for source in out["sources"]}
    assert out["passages"] and all(p["text"] == snippets[p["index"]] for p in out["passages"])


def test_ground_errors_cancellation_and_fallbacks(site: Site) -> None:
    with pytest.raises(BadRequest) as forbidden:
        WebSearchEngine(f"{site.base}/forbidden").ground("solar panels")
    assert "json" in (forbidden.value.hint or "").lower()
    with pytest.raises(BadRequest):
        WebSearchEngine("").ground("solar panels")

    engine = WebSearchEngine(site.base, allow_private_hosts=True)
    stopped = threading.Event()
    stopped.set()
    with pytest.raises(Cancelled):
        engine.ground("solar panels", stop_event=stopped)
    stop = threading.Event()
    with pytest.raises(Cancelled):
        engine.ground("solar panels", stop_event=stop,
                      on_status=lambda message: stop.set() if message.startswith("Reading") else None)

    def broken(_question: str) -> list[str]:
        raise RuntimeError("model crashed")

    out = engine.ground("solar panels efficiency", generate_queries=broken)
    assert out["queries"] == ["solar panels efficiency"] and out["sources"]
    assert engine.ground("   ") == {"queries": [], "sources": [], "passages": [], "context": ""}
