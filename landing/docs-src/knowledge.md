---
title: Knowledge
nav: Knowledge
description: Build local knowledge bases from your documents, chat with them with citations and measure retrieval quality.
section: Using Noelle
order: 13
slug: knowledge
---

**Knowledge** turns your documents into searchable knowledge bases that chat models can use to answer with
citations. Everything — reading files, embeddings and search — happens on your computer.

## Create a knowledge base

1. Open **Knowledge** and click **New knowledge base**.
2. Give it a name and pick an **embedding model**. For English, `sentence-transformers/all-MiniLM-L6-v2` is small and
   fast; for Indonesian and other languages, `BAAI/bge-m3` works well. If the model isn't downloaded yet, Noelle
   offers to download it.
3. On the **Documents** tab, add files or a whole folder.

Supported files include PDF, Word (`.docx`), Markdown, plain text, HTML, CSV, JSON and source code. Hidden folders
such as `.git` are skipped. Scanned PDFs without a text layer are reported as skipped.

## Search

The **Search** tab runs a query against the base and shows the best passages with their score, file and page, and
a button to copy a citation.

## Chat with your documents

In the Workspace text playground, tick **Documents** and pick a knowledge base. Answers cite their sources, and the
**Sources** panel shows the passages that were used. Tick **Web search** as well to add results from your own
[SearXNG](https://docs.searxng.org/) instance (set its address in **Settings › Grounding**).

## Measure retrieval quality

The **Evaluate** tab checks whether the right documents come back for your questions. Add cases — a question and
the file names (or parts of paths) that should be found — or import them as JSONL:

```json
{"query": "How do I reset the router?", "relevant": ["router-manual.pdf"]}
{"query": "What is the refund period?", "relevant": ["policies/refunds.md"]}
```

Pick **k** (how many results count) and run. Noelle reports:

| Metric | Meaning |
|---|---|
| **Hit rate@k** | Share of questions with at least one relevant document in the top k |
| **Recall@k** | Share of the expected documents that were found |
| **Precision@k** | Share of the top-k results that are relevant |
| **MRR** | How high the first relevant result ranks, on average |
| **nDCG@k** | Ranking quality, rewarding relevant results near the top |

Select a second knowledge base to compare **A vs B** — for example the same documents with two chunk sizes or two
embedding models.

## Settings

Each knowledge base has its own **chunk size** and **overlap** (512 and 64 tokens by default), used for documents
you add afterwards. The embedding model is fixed once the base contains documents.
