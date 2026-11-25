"""Command-line dispatcher.

``noelle``            launch the desktop studio (default)
``noelle worker``     run the inference worker (spawned by the studio)
``noelle server``     run the OpenAI-compatible API server (spawned by the studio)
``noelle download``   download a Hub repository with JSON progress on stdout
``noelle env``        manage isolated runtime environments
``noelle mcp``        Model Context Protocol server on stdio (for Claude Code, Cursor, ...)
``noelle eval``       Evaluation Lab from the terminal (thresholds/baselines for CI)
``noelle analyze``    compatibility score and run options for a Hub repository
``noelle run``        one prompt to a local model, streamed to stdout

A frozen (PyInstaller) build re-executes itself with these subcommands, so every
child process is the same binary.
"""

from __future__ import annotations

import argparse
import sys

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="noelle", description="Noelle Desktop AI Studio")
    sub = parser.add_subparsers(dest="command", metavar="{ui,worker,server,download,setup,eval,analyze,run,mcp,env}")
    sub.add_parser("ui", help="launch the desktop studio")

    from noelle.runtime import worker_service

    worker = sub.add_parser("worker", help="run the inference worker")
    worker_service.add_arguments(worker)

    server = sub.add_parser("server", help="run the OpenAI-compatible API server")
    server.add_argument("--host", default="127.0.0.1")
    server.add_argument("--port", type=int, default=8765)
    server.add_argument("--worker-endpoint", default=None, help="tcp://127.0.0.1:<port> (default: discovery file)")
    server.add_argument("--parent-pid", type=int, default=None, help="exit when this process exits")

    download = sub.add_parser("download", help="download a Hub repository (JSON-lines progress)")
    download.add_argument("repo_id")
    download.add_argument("--revision", default=None)
    download.add_argument("--repo-type", default="model")
    download.add_argument("--local-dir", default=None)
    download.add_argument("--include", action="append", default=None, help="allow pattern (repeatable)")
    download.add_argument("--exclude", action="append", default=None, help="ignore pattern (repeatable)")
    download.add_argument("--max-workers", type=int, default=8)
    download.add_argument("--retries", type=int, default=5)
    download.add_argument("--no-verify", action="store_true")
    download.add_argument("--dry-run", action="store_true")

    setup = sub.add_parser("setup", help="open the setup assistant, or import a token (used by installers)")
    setup.add_argument("--import-hf-token", metavar="FILE", default=None,
                       help="read a Hugging Face token from FILE into the OS credential store, then delete FILE")

    from noelle import lab_cli

    lab_cli.add_eval_arguments(sub.add_parser("eval", help="evaluate local models on datasets (CI-friendly)"))
    lab_cli.add_analyze_arguments(sub.add_parser("analyze", help="score how well a Hub model runs on this machine"))
    lab_cli.add_run_arguments(sub.add_parser("run", help="send one prompt to a local model and stream the reply"))

    mcp = sub.add_parser("mcp", help="run the Model Context Protocol server on stdio")
    mcp.add_argument("--api-url", default=None,
                     help="Noelle API server base URL (default: http://<server.host>:<server.port> from settings)")
    mcp.add_argument("--print-config", action="store_true",
                     help="print a JSON snippet for an MCP client's configuration and exit")

    pyrun = sub.add_parser("pyrun")  # frozen builds: run a helper script (GGUF converter); hidden from --help
    pyrun.add_argument("script")
    pyrun.add_argument("script_args", nargs=argparse.REMAINDER)
    sub._choices_actions = [a for a in sub._choices_actions if a.dest != "pyrun"]

    env = sub.add_parser("env", help="manage isolated runtime environments")
    env_sub = env.add_subparsers(dest="env_command")
    create = env_sub.add_parser("create")
    create.add_argument("name")
    create.add_argument("--profile", action="append", default=None)
    create.add_argument("--python", default=None)
    env_sub.add_parser("list")
    remove = env_sub.add_parser("remove")
    remove.add_argument("name")
    verify = env_sub.add_parser("verify")
    verify.add_argument("name")
    export = env_sub.add_parser("export", help="write a pinned lockfile (noelle-env.yaml)")
    export.add_argument("name")
    export.add_argument("--output", "-o", default=None)
    lock_import = env_sub.add_parser("import", help="create an environment from a lockfile")
    lock_import.add_argument("file")
    lock_import.add_argument("--name", default=None)
    lock_import.add_argument("--python", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    commands = {"ui", "worker", "server", "download", "env", "setup", "pyrun", "mcp", "eval", "analyze", "run",
                "-h", "--help"}
    if not argv or argv[0] not in commands:
        # Default to the UI; leave Qt's own arguments (e.g. -platform offscreen) for QApplication.
        from noelle.app.main import run

        return run(argv)
    if argv[0] not in ("worker", "server", "download", "pyrun", "ui", "setup"):
        _attach_windows_console()
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "ui":
        from noelle.app.main import run

        return run([])
    if args.command == "worker":
        from noelle.runtime import worker_service

        return worker_service.main(args)
    if args.command == "server":
        from noelle.runtime import server_service

        return server_service.main(args)
    if args.command == "download":
        from noelle.runtime import downloader

        return downloader.main(args)
    if args.command == "setup":
        if args.import_hf_token:
            return import_hf_token(args.import_hf_token)
        from noelle.app.main import run

        return run(["--setup"])
    if args.command in ("eval", "analyze", "run"):
        from noelle import lab_cli

        handler = {"eval": lab_cli.run_eval, "analyze": lab_cli.run_analyze, "run": lab_cli.run_prompt}[args.command]
        return handler(args)
    if args.command == "mcp":
        from noelle.runtime import mcp_server

        return mcp_server.main(args)
    if args.command == "pyrun":
        import runpy
        from pathlib import Path

        sys.argv = [args.script, *args.script_args]
        sys.path.insert(0, str(Path(args.script).resolve().parent))
        runpy.run_path(args.script, run_name="__main__")
        return 0
    if args.command == "env":
        from noelle.runtime import environment_manager

        return environment_manager.main(args)
    parser.print_help()
    return 1


def _attach_windows_console() -> None:
    """Windowed (GUI) Windows builds start without a console: borrow the terminal that launched them so
    ``Noelle.exe eval …`` prints. Pipes from a parent process (e.g. an MCP client) are left alone."""
    if sys.platform != "win32" or not getattr(sys, "frozen", False) or sys.stdout is not None:
        return
    import ctypes

    if ctypes.windll.kernel32.AttachConsole(-1):  # ATTACH_PARENT_PROCESS
        sys.stdout = open("CONOUT$", "w", encoding="utf-8", buffering=1)  # noqa: SIM115 - process lifetime
        sys.stderr = open("CONOUT$", "w", encoding="utf-8", buffering=1)  # noqa: SIM115
        sys.stdin = open("CONIN$", encoding="utf-8")  # noqa: SIM115


def import_hf_token(path: str) -> int:
    """Store a token handed over by an installer, then delete the hand-over file."""
    from pathlib import Path

    from noelle.app.services import hf_client

    file = Path(path)
    try:
        token = file.read_text(encoding="utf-8").strip()
    except OSError as exc:
        print(f"Could not read {file}: {exc}", file=sys.stderr)
        return 1
    finally:
        try:
            file.unlink()
        except OSError:
            pass
    if not token:
        return 0
    try:
        hf_client.set_token(token)
    except Exception as exc:  # no keyring backend
        print(f"Could not store the token in the credential store: {exc}", file=sys.stderr)
        return 1
    print("Hugging Face token saved to the system credential store.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
