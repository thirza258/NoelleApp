# Noelle website — noelle.nevatal.id

A static site: the landing page, the documentation and a site map. No framework and no build step for
hosting — upload this folder as it is.

```
landing/
├── index.html          landing page
├── config.js           ← paste your SourceForge links here
├── styles.css          light (white) and dark (neutral grey) themes, like the app
├── main.js             download links, theme toggle, copy buttons, mobile menu
├── assets/             icon, favicon, app screenshots (light + dark)
├── docs/               generated documentation pages (do not edit by hand)
├── docs-src/           documentation sources (Markdown) — edit these
├── build.py            regenerates docs/, sitemap.xml, robots.txt, 404.html and the footer site map
├── sitemap.xml, robots.txt, 404.html, CNAME
```

## 1. Add the SourceForge link

Open `config.js` and replace `YOUR-PROJECT` with your SourceForge project name:

```js
window.NOELLE_LINKS = {
  download: "https://sourceforge.net/projects/noelle/files/latest/download",
  files: "https://sourceforge.net/projects/noelle/files/",
  source: "https://github.com/thirza258/NoelleApp",
};
```

Every Download button on the landing page and in the docs uses these links. SourceForge's
`/files/latest/download` link picks the right installer for the visitor's operating system.

## 2. Preview

```bash
python3 -m http.server 8000 --directory landing
# open http://localhost:8000
```

## 3. Edit the documentation

Docs pages are Markdown files in `docs-src/` with a short header:

```markdown
---
title: Evaluation Lab
nav: Evaluation Lab            # name in the sidebar (optional)
description: One sentence for search results and the page intro.
section: Using Noelle          # Get started · Using Noelle · Developers · Reference
order: 12                      # position inside the section
slug: evaluation               # URL: /docs/evaluation/
---
```

Link to other pages with `docs:<slug>` (`[the API](docs:api)`, `[trust](docs:models#repositories-that-run-their-own-code)`)
and to the download buttons with `download:`, `files:` or `source:`. Then regenerate:

```bash
pip install markdown-it-py
python3 landing/build.py
```

`build.py` rewrites `docs/`, `sitemap.xml`, `robots.txt`, `404.html`, `CNAME` and the site-map footer of
`index.html` (between the `sitemap:start` / `sitemap:end` markers). It stops with an error if a `docs:` link points to a
page that doesn't exist. The site address is `BASE_URL` at the top of `build.py`; if it ever changes, update it there
and in the `<head>` of `index.html` (canonical and `og:` URLs).

## 4. Publish on noelle.nevatal.id

The pages use relative links, so any static host works. Two common options:

- **GitHub Pages:** publish the `landing/` folder (for example with the *Deploy static content to Pages* workflow,
  path `landing`). The `CNAME` file sets the custom domain. At your DNS provider for `nevatal.id`, add a
  `CNAME` record `noelle` → `<your-github-user>.github.io`, then enable **Enforce HTTPS** in the repository's Pages
  settings.
- **Any web server or CDN:** upload the folder's contents to the site root and point `noelle.nevatal.id` at it.
  Configure `404.html` as the not-found page.

After the site is live, submit `https://noelle.nevatal.id/sitemap.xml` in Google Search Console and Bing Webmaster
Tools (`robots.txt` already points crawlers to it).

## Screenshots

`assets/screens/*-light.png` and `*-dark.png` are real screenshots of the app at 1440×900. Replace them with
new captures of the same size and names; the page shows the dark version when the dark theme is active.
