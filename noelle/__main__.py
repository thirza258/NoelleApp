"""Command-line dispatcher.

``noelle``            launch the desktop studio (default)
``noelle worker``     run the inference worker (spawned by the studio)
``noelle server``     run the OpenAI-compatible API server (spawned by the studio)
``noelle download``   download a Hub repository with JSON progress on stdout
``noelle env``        manage isolated runtime environments

A frozen (PyInstaller) build re-executes itself with these subcommands, so every
child process is the same binary.
"""

from __future__ import annotations

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="noelle", description="Noelle Desktop AI Studio")
    sub = parser.add_subparsers(dest="command")
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
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    commands = {"ui", "worker", "server", "download", "env", "-h", "--help"}
    if not argv or argv[0] not in commands:
        # Default to the UI; leave Qt's own arguments (e.g. -platform offscreen) for QApplication.
        from noelle.app.main import run

        return run(argv)
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
    if args.command == "env":
        from noelle.runtime import environment_manager

        return environment_manager.main(args)
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
