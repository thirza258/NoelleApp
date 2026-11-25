# Noelle logging

All logs live under the data root's `logs/` folder (`~/.noelle/logs` by default; set
`NOELLE_HOME` to move it). Files are size-capped and rotated, so a long session cannot fill the
disk. Every line includes the timestamp, level, logger, process id and thread name.

## Files

| File | Written by | Contents |
| --- | --- | --- |
| `app.log` | desktop shell | overview of everything at INFO and above |
| `errors.log` | desktop shell | every ERROR/CRITICAL record (quick triage) |
| `ipc.log` | desktop shell | worker start/ready/crash/restart, request refusals, IPC failures |
| `downloads.log` | desktop shell | download queue, launches, retries, completion/failure |
| `download-child.log` | download child | per-file download, checksum verification, retries |
| `credentials.log` | desktop shell | OS credential-store operations (never the token) |
| `hub.log` | desktop shell | Hugging Face Hub metadata requests and failures |
| `storage.log` | desktop shell | registry changes, cache pruning, relocation |
| `models.log` | worker | model load/unload/eviction, memory estimates and timings |
| `worker.log` | worker | job queue, execution, cancellation, failures |
| `server.log` | API server | startup/shutdown and server errors |
| `server-access.log` | API server | one line per HTTP request (no prompt/completion bodies) |
| `server-manager.log` | desktop shell | API server child supervision |
| `mcp.log` | MCP server | stdio tool calls and failures (protocol stays on stdout) |
| `web.log`, `rag.log`, `quantize.log` | worker | web search, retrieval, quantization |
| `tasks.log` | desktop shell | background UI task failures with tracebacks |
| `ui.log` | desktop shell | UI warnings, notebook lifecycle, shutdown |
| `settings.log` | desktop shell | settings reload/save problems |
| `setup.log` | installer/import | Hugging Face token hand-over result (never the token) |
| `debug.log` | desktop shell | optional verbose copy (see below) |

## Verbosity

- Default: `app.log` and process logs at INFO; dedicated component files capture DEBUG detail.
- `NOELLE_LOG_LEVEL=DEBUG` raises the main file/process log to DEBUG.
- `NOELLE_LOG_DEBUG=1` additionally writes a complete `debug.log` for the desktop shell.
- Invalid values fall back to the default; logging never prevents startup.

## Security

A formatting filter redacts credential-shaped strings (Hugging Face `hf_…` tokens, `sk-…` API
keys and `Bearer …` headers) from messages, arguments and traceback text before anything is
written. API keys, prompts and generated content are not logged by design; the API access log
records only method, path, status and duration.

## Support

Settings → About has "Open logs", which opens this folder. When reporting a problem, include
`errors.log`, `ipc.log` (runtime problems), `downloads*.log` (downloads) or the relevant
subsystem file. Logs contain no credentials and can be shared safely.