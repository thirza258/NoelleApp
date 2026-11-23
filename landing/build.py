#!/usr/bin/env python3
"""Build the Noelle website's documentation and site map.

    pip install markdown-it-py      # once
    python3 landing/build.py        # after editing landing/docs-src/*.md

Writes (all committed, so hosting needs no build step):

* ``docs/index.html`` and ``docs/<slug>/index.html`` from ``docs-src/*.md``
* ``sitemap.xml`` and ``robots.txt`` for search engines
* ``404.html``
* the site-map footer inside ``index.html`` (between the ``sitemap:start`` / ``sitemap:end`` markers)

Docs pages are Markdown with a small front-matter block (title, nav, description, section, order, slug).
Links between pages use ``docs:<slug>`` (optionally ``#anchor``); ``download:``, ``files:`` and ``source:``
become buttons that use the URLs from ``config.js``.
"""

from __future__ import annotations

import datetime as _dt
import html
import re
import sys
from dataclasses import dataclass
from pathlib import Path

BASE_URL = "https://noelle.nevatal.id"
SITE = Path(__file__).resolve().parent
SOURCES = SITE / "docs-src"
SECTIONS = ("Get started", "Using Noelle", "Developers", "Reference")


@dataclass
class Page:
    slug: str
    title: str
    nav: str
    description: str
    section: str
    order: int
    body: str
    source: Path

    @property
    def path(self) -> str:
        """Site path, e.g. ``docs/install/``."""
        return "docs/" if not self.slug else f"docs/{self.slug}/"

    @property
    def depth(self) -> int:
        return self.path.count("/")


def parse(path: Path) -> Page:
    text = path.read_text(encoding="utf-8")
    match = re.match(r"^---\n(.*?)\n---\n", text, re.DOTALL)
    if not match:
        raise SystemExit(f"{path.name}: missing front matter")
    meta: dict[str, str] = {}
    for line in match.group(1).splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            meta[key.strip()] = value.strip().strip('"').strip("'")
    for key in ("title", "description", "section", "order"):
        if key not in meta:
            raise SystemExit(f"{path.name}: front matter needs '{key}'")
    if meta["section"] not in SECTIONS:
        raise SystemExit(f"{path.name}: unknown section {meta['section']!r} (use one of {SECTIONS})")
    return Page(slug=meta.get("slug", path.stem), title=meta["title"], nav=meta.get("nav") or meta["title"],
                description=meta["description"], section=meta["section"], order=int(meta["order"]),
                body=text[match.end():], source=path)


def slugify(text: str) -> str:
    text = html.unescape(re.sub(r"<[^>]+>", "", text)).lower()
    return re.sub(r"[^a-z0-9]+", "-", text).strip("-") or "section"


def relative(from_page: Page | None, target: str) -> str:
    """Relative URL from a page (None = site root) to a site path such as ``docs/install/``."""
    up = "../" * (from_page.depth if from_page else 0)
    return (up + target) or "./"


def render_markdown(page: Page, pages: dict[str, Page]) -> str:
    try:
        from markdown_it import MarkdownIt
    except ImportError:
        raise SystemExit("markdown-it-py is required: pip install markdown-it-py") from None
    body = MarkdownIt("commonmark", {"html": True}).enable("table").enable("strikethrough").render(page.body)

    def docs_link(match: re.Match[str]) -> str:
        slug, anchor = match.group(1), match.group(2) or ""
        if slug not in pages:
            raise SystemExit(f"{page.source.name}: link to unknown page docs:{slug}")
        return f'href="{relative(page, pages[slug].path)}{anchor}"'

    body = re.sub(r'href="docs:([a-z0-9-]*)(#[^"]*)?"', docs_link, body)
    body = re.sub(r'href="(download|files|source):"',
                  lambda m: f'href="{relative(page, "")}#download" data-link="{m.group(1)}"', body)
    body = re.sub(r'<a href="(https?://[^"]+)"', r'<a href="\1" rel="noopener"', body)

    seen: set[str] = set()

    def heading(match: re.Match[str]) -> str:
        level, inner = match.group(1), match.group(2)
        anchor = slugify(inner)
        base, n = anchor, 2
        while anchor in seen:
            anchor, n = f"{base}-{n}", n + 1
        seen.add(anchor)
        return (f'<h{level} id="{anchor}">{inner}<a class="heading-anchor" href="#{anchor}" '
                f'aria-label="Link to this section">#</a></h{level}>')

    body = re.sub(r"<h([23])>(.*?)</h\1>", heading, body)
    body = body.replace("<table>", '<table class="table">')
    return body


# ---------------------------------------------------------------------------
# shared layout
# ---------------------------------------------------------------------------

ICON_SPRITE = (SITE / "index.html").read_text(encoding="utf-8").split("<defs>", 1)[1].split("</defs>", 1)[0]


def head(title: str, description: str, url: str, prefix: str, root_links: bool = False) -> str:
    asset = "/" if root_links else prefix
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <meta name="description" content="{html.escape(description)}">
  <link rel="canonical" href="{url}">
  <meta property="og:type" content="article">
  <meta property="og:title" content="{html.escape(title)}">
  <meta property="og:description" content="{html.escape(description)}">
  <meta property="og:url" content="{url}">
  <meta property="og:image" content="{BASE_URL}/assets/screens/mission-light.png">
  <meta name="twitter:card" content="summary_large_image">
  <meta name="theme-color" content="#FFFFFF" media="(prefers-color-scheme: light)">
  <meta name="theme-color" content="#1E1E1E" media="(prefers-color-scheme: dark)">
  <link rel="icon" type="image/png" href="{asset}assets/favicon.png">
  <link rel="stylesheet" href="{asset}styles.css">
  <script>(function () {{ var t = null; try {{ t = localStorage.getItem("noelle-theme"); }} catch (e) {{}}
    if (t === "light" || t === "dark") document.documentElement.setAttribute("data-theme", t); }})();</script>
  <script src="{asset}config.js"></script>
  <script src="{asset}main.js" defer></script>
</head>
<body>
  <svg width="0" height="0" style="position:absolute" aria-hidden="true"><defs>{ICON_SPRITE}</defs></svg>
"""


def header(prefix: str, current: str = "") -> str:
    def nav(href: str, text: str, key: str) -> str:
        mark = ' aria-current="page"' if key == current else ""
        return f'<a href="{href}"{mark}>{text}</a>'

    return f"""  <header class="site-header">
    <div class="container nav">
      <a class="brand" href="{prefix or './'}" aria-label="Noelle home">
        <img src="{prefix}assets/noelle.png" alt="" width="30" height="30"><span>Noelle</span>
      </a>
      <nav class="nav-links" aria-label="Main">
        {nav(prefix + "#features", "Features", "features")}
        {nav(prefix + "docs/", "Docs", "docs")}
        {nav(prefix + "docs/quickstart/", "Quick start", "quickstart")}
        {nav(prefix + "docs/api/", "API", "api")}
        {nav(prefix + "#faq", "FAQ", "faq")}
      </nav>
      <div class="nav-actions">
        <button class="icon-btn menu-btn" type="button" data-menu aria-label="Menu" aria-expanded="false">
          <svg class="icon"><use href="#i-menu"/></svg></button>
        <button class="icon-btn" type="button" data-theme-toggle aria-label="Switch between light and dark">
          <svg class="icon show-light"><use href="#i-moon"/></svg><svg class="icon show-dark"><use href="#i-sun"/></svg>
        </button>
        <a class="btn btn-primary btn-small" href="{prefix}#download" data-link="download">
          <svg class="icon"><use href="#i-download"/></svg><span>Download</span></a>
      </div>
    </div>
  </header>
"""


def sitemap_footer(pages: list[Page], from_page: Page | None) -> str:
    """The footer site map: every page of the site, grouped like the docs sidebar."""
    def link(target: str, text: str, extra: str = "") -> str:
        return f'<li><a href="{relative(from_page, target)}{extra}">{html.escape(text)}</a></li>'

    columns = {
        "Product": [link("", "Features", "#features"), link("", "How it works", "#how"),
                    link("", "Privacy", "#privacy"), link("", "Developers", "#developers"),
                    link("", "Download", "#download"), link("", "FAQ", "#faq")],
        "Documentation": [link(p.path, p.nav) for p in pages if p.section in ("Get started", "Using Noelle")],
        "Developers & reference": [link(p.path, p.nav) for p in pages if p.section in ("Developers", "Reference")],
    }
    cols = "\n".join(f'        <div><h4>{name}</h4><ul>{"".join(items)}</ul></div>' for name, items in columns.items())
    home = relative(from_page, "")
    return f"""  <footer class="site-footer">
    <div class="container">
      <div class="footer-grid">
        <div class="footer-about">
          <a class="brand" href="{home}"><img src="{relative(from_page, 'assets/noelle.png')}" alt="" width="24" height="24"><span>Noelle</span></a>
          <p>Your local AI model lab — find, run, evaluate and serve open models on your own computer.</p>
        </div>
{cols}
      </div>
      <div class="footer-bottom">
        <span>© <span data-year></span> Noelle · MIT License</span>
        <span><a href="{relative(from_page, 'docs/privacy/')}">Privacy Policy</a> ·
          <a href="{relative(from_page, 'docs/terms/')}">Terms of Service</a> ·
          <a href="{home}#download" data-link="source">Source code</a> ·
          <a href="{relative(from_page, 'sitemap.xml')}">sitemap.xml</a></span>
      </div>
    </div>
  </footer>
"""


def sidebar(pages: list[Page], current: Page) -> str:
    parts = ['<aside class="docs-nav" aria-label="Documentation">']
    for section in SECTIONS:
        items = [p for p in pages if p.section == section]
        if not items:
            continue
        parts.append(f"<h4>{section}</h4><ul>")
        for p in items:
            mark = ' aria-current="page"' if p is current else ""
            parts.append(f'<li><a href="{relative(current, p.path)}"{mark}>{html.escape(p.nav)}</a></li>')
        parts.append("</ul>")
    parts.append("</aside>")
    return "".join(parts)


def pager(pages: list[Page], current: Page) -> str:
    index = pages.index(current)
    prev_page = pages[index - 1] if index > 0 else None
    next_page = pages[index + 1] if index + 1 < len(pages) else None
    out = ['<nav class="pager" aria-label="Previous and next page">']
    if prev_page:
        out.append(f'<a class="prev" href="{relative(current, prev_page.path)}"><span>Previous</span>'
                   f"<b>{html.escape(prev_page.nav)}</b></a>")
    if next_page:
        out.append(f'<a class="next" href="{relative(current, next_page.path)}"><span>Next</span>'
                   f"<b>{html.escape(next_page.nav)}</b></a>")
    out.append("</nav>")
    return "".join(out)


def render_page(page: Page, pages: list[Page], by_slug: dict[str, Page]) -> str:
    prefix = "../" * page.depth
    title = "Noelle documentation" if not page.slug else f"{page.title} — Noelle docs"
    return (head(title, page.description, f"{BASE_URL}/{page.path}", prefix)
            + header(prefix, "quickstart" if page.slug == "quickstart" else "api" if page.slug == "api" else "docs")
            + f"""  <main class="container docs" id="top">
    {sidebar(pages, page)}
    <article class="prose">
      <h1>{html.escape(page.title)}</h1>
      <p class="page-lead">{html.escape(page.description)}</p>
{render_markdown(page, by_slug)}
      {pager(pages, page)}
    </article>
  </main>
""" + sitemap_footer(pages, page) + "</body>\n</html>\n")


def render_404(pages: list[Page]) -> str:
    body = """  <main class="container notfound">
    <h1>Page not found</h1>
    <p class="lead" style="margin-inline:auto">That page doesn't exist — it may have moved.</p>
    <p><a class="btn btn-primary" href="/">Go to the home page</a> <a class="btn btn-ghost" href="/docs/">Read the docs</a></p>
  </main>
"""
    # a 404 page is served at any path, so its links must be absolute
    footer = sitemap_footer(pages, None).replace('href="./', 'href="/').replace('href="docs/', 'href="/docs/')
    footer = footer.replace('href="sitemap.xml"', 'href="/sitemap.xml"').replace('src="assets/', 'src="/assets/')
    return head("Page not found — Noelle", "This page doesn't exist.", f"{BASE_URL}/404.html", "/", True) + \
        header("/") + body + footer + "</body>\n</html>\n"


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.read_text(encoding="utf-8") != text:
        path.write_text(text, encoding="utf-8")
        print(f"wrote {path.relative_to(SITE)}")


def main() -> int:
    pages = sorted((parse(p) for p in SOURCES.glob("*.md")),
                   key=lambda p: (SECTIONS.index(p.section), p.order, p.slug))
    by_slug = {p.slug: p for p in pages}
    if "" not in by_slug:
        raise SystemExit("docs-src needs an overview page with slug: \"\"")
    for page in pages:
        write(SITE / page.path / "index.html", render_page(page, pages, by_slug))

    landing = SITE / "index.html"
    text = landing.read_text(encoding="utf-8")
    start, end = "<!-- sitemap:start -->", "<!-- sitemap:end -->"
    if start not in text or end not in text:
        raise SystemExit("index.html needs the <!-- sitemap:start --> / <!-- sitemap:end --> markers")
    before, rest = text.split(start, 1)
    after = rest.split(end, 1)[1]
    write(landing, before + start + "\n" + sitemap_footer(pages, None) + "  " + end + after)
    write(SITE / "404.html", render_404(pages))

    today = _dt.date.today().isoformat()
    urls = [("", "1.0", today)] + [(p.path, "0.8" if p.slug else "0.9",
                                    _dt.date.fromtimestamp(p.source.stat().st_mtime).isoformat()) for p in pages]
    entries = "\n".join(f"  <url>\n    <loc>{BASE_URL}/{path}</loc>\n    <lastmod>{mod}</lastmod>\n"
                        f"    <priority>{priority}</priority>\n  </url>" for path, priority, mod in urls)
    write(SITE / "sitemap.xml", '<?xml version="1.0" encoding="UTF-8"?>\n'
          '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n' + entries + "\n</urlset>\n")
    write(SITE / "robots.txt", f"User-agent: *\nAllow: /\n\nSitemap: {BASE_URL}/sitemap.xml\n")
    write(SITE / "CNAME", BASE_URL.split("://", 1)[1] + "\n")
    print(f"{len(pages)} docs pages · {len(urls)} URLs in sitemap.xml")
    return 0


if __name__ == "__main__":
    sys.exit(main())
