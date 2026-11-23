---
title: MCP server
nav: MCP server
description: Let Claude Code, Cursor and other agents use your local models, knowledge bases and Hugging Face search through the Model Context Protocol.
section: Developers
order: 21
slug: mcp
---

`noelle mcp` runs a [Model Context Protocol](https://modelcontextprotocol.io) server, so coding agents such as
Claude Code and Cursor can use Noelle as a set of tools.

## Set it up

**Claude Code:**

```bash
claude mcp add noelle -- noelle mcp
```

**Other clients (Cursor, Claude Desktop, …):** run `noelle mcp --print-config` and paste the JSON it prints into
the client's MCP configuration. It contains the right command for your installation, for example:

```json
{
  "mcpServers": {
    "noelle": { "command": "/Applications/Noelle.app/Contents/MacOS/Noelle", "args": ["mcp"] }
  }
}
```

The `noelle` command is on your path after installing the Linux package. On macOS use
`/Applications/Noelle.app/Contents/MacOS/Noelle`, and on Windows the `Noelle.exe` in your installation folder.

## Tools

| Tool | What it does |
|---|---|
| `hardware_info` | Your accelerator, usable memory, memory bandwidth and installed engines |
| `list_local_models` | Models downloaded or built in Noelle |
| `search_models` | Hugging Face search with a fit score for this machine |
| `inspect_model` | Everything in the Analysis tab: run options, recommendation, context, speed |
| `chat` | A chat completion with a local model (`"model": "auto"` lets Noelle choose) |
| `embed` | Embedding vectors from a local embedding model |
| `list_knowledge_bases`, `query_knowledge_base` | Search your knowledge bases |

`chat`, `embed` and the knowledge tools go through the [local API](docs:api), so **start the server** on the API page
first. If you set an API key, the MCP server reads it from Noelle's settings.

## Privacy

The MCP server talks only to the API server on this computer — it refuses any other address — and never logs your
prompts or the API key. It cannot download, delete or run anything outside these tools.
