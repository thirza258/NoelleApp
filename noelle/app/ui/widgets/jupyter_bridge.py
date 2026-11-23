"""Notebook tab: an embedded Jupyter console bridged to a local kernel.

The kernel is its own process (like the worker), so importing torch there never
touches the desktop shell. It is pre-seeded with ``noelle.load_model(...)`` for
the current model.
"""

from __future__ import annotations

import importlib.util
import os
from typing import Any

from PySide6.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget

from noelle.app import theme
from noelle.app.context import AppContext
from noelle.app.ui.widgets.common import Banner, EmptyState, button, label
from noelle.runtime.client import build_worker_env


def jupyter_available() -> bool:
    return all(importlib.util.find_spec(name) is not None for name in ("qtconsole", "ipykernel", "jupyter_client"))


class NotebookPanel(QWidget):
    def __init__(self, ctx: AppContext, page: Any) -> None:
        super().__init__()
        self.ctx = ctx
        self.page = page
        self.kernel_manager: Any = None
        self.kernel_client: Any = None
        self.console: Any = None
        root = QVBoxLayout(self)
        root.setContentsMargins(18, 14, 18, 14)
        root.setSpacing(10)
        head = QHBoxLayout()
        titles = QVBoxLayout()
        titles.addWidget(label("Notebook", "h2"))
        titles.addWidget(label("A live Python kernel for cell-by-cell experiments. It runs in its own process with "
                               "its own copy of the model.", "muted", wrap=True))
        head.addLayout(titles, 1)
        self.start_btn = button("Start kernel", "play", "primary")
        self.start_btn.clicked.connect(self.start_kernel)
        head.addWidget(self.start_btn)
        self.snippet_btn = button("Insert load snippet", "code", "ghost")
        self.snippet_btn.clicked.connect(self.insert_snippet)
        self.snippet_btn.setEnabled(False)
        head.addWidget(self.snippet_btn)
        self.stop_btn = button("Stop", "stop", "ghost")
        self.stop_btn.clicked.connect(self.shutdown)
        self.stop_btn.setEnabled(False)
        head.addWidget(self.stop_btn)
        root.addLayout(head)
        self.memory_note = Banner("info", "Loading the model in the notebook uses extra memory on top of the runtime. "
                                          "Unload it from the runtime first if memory is tight.")
        root.addWidget(self.memory_note)
        self.body = QVBoxLayout()
        root.addLayout(self.body, 1)
        if not jupyter_available():
            self.start_btn.setEnabled(False)
            self.body.addWidget(EmptyState("terminal", "Jupyter is not installed",
                                           'Install the extra with: pip install "noelle[jupyter]"'))

    def _snippet(self) -> str:
        payload = self.page.worker_payload() if self.page.summary else {}
        model_id = payload.get("model_id", "")
        args = [repr(model_id)]
        if payload.get("task"):
            args.append(f"task={payload['task']!r}")
        if payload.get("precision") and payload["precision"] != "auto":
            args.append(f"precision={payload['precision']!r}")
        return ("import noelle.sdk as noelle\n"
                f"model = noelle.load_model({', '.join(args)})\n"
                "model")

    def start_kernel(self) -> None:
        if self.kernel_manager is not None or not jupyter_available():
            return
        from qtconsole.manager import QtKernelManager
        from qtconsole.rich_jupyter_widget import RichJupyterWidget

        env = build_worker_env(None, hf_home=self.ctx.settings.get("storage.hf_home"), offline=True)
        env["NOELLE_HOME"] = os.environ.get("NOELLE_HOME", env.get("NOELLE_HOME", ""))
        if not env["NOELLE_HOME"]:
            env.pop("NOELLE_HOME")
        manager = QtKernelManager(kernel_name="python3")
        manager.start_kernel(env=env)
        client = manager.client()
        client.start_channels()
        console = RichJupyterWidget()
        console.kernel_manager = manager
        console.kernel_client = client
        console.set_default_style(colors="linux" if theme.IS_DARK else "lightbg")
        console.setStyleSheet(f"background: {theme.CODE_BG};")
        self.kernel_manager, self.kernel_client, self.console = manager, client, console
        self.body.addWidget(console, 1)
        self.start_btn.setEnabled(False)
        self.snippet_btn.setEnabled(True)
        self.stop_btn.setEnabled(True)
        console._append_plain_text("Kernel ready. 'Insert load snippet' loads the current model.\n", before_prompt=True)

    def insert_snippet(self) -> None:
        if self.console is not None:
            self.console.input_buffer = self._snippet()

    def shutdown(self) -> None:
        if self.kernel_client is not None:
            try:
                self.kernel_client.stop_channels()
            except Exception:
                pass
        if self.kernel_manager is not None:
            try:
                self.kernel_manager.shutdown_kernel(now=True)
            except Exception:
                pass
        if self.console is not None:
            self.console.setParent(None)
            self.console.deleteLater()
        self.kernel_manager = self.kernel_client = self.console = None
        self.start_btn.setEnabled(jupyter_available())
        self.snippet_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)

