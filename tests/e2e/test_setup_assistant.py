"""Setup Assistant: token goes to the keychain, nothing downloads without a tick, completion is remembered."""

from __future__ import annotations

import os
from typing import Any

import pytest

pytestmark = pytest.mark.e2e
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture()
def ctx(qapp: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
    from noelle.app import main as appmain
    from noelle.app.services import components

    monkeypatch.setattr(components, "resolve_llama_cpp", lambda session=None: components.ComponentRelease(
        tag="b1", asset_name="llama-b1-bin-macos-arm64.tar.gz", asset_url="https://x", asset_size=16_600_000,
        asset_sha256="ab", source_url="https://y"))
    installs: list[Any] = []
    monkeypatch.setattr(components, "install_llama_cpp", lambda *a, **k: installs.append(a) or {"tag": "b1"})
    context = appmain.build_context()
    context.extras["installs"] = installs
    return context


def test_assistant_flow_without_downloads(ctx: Any, qtbot: Any, memory_keyring: Any) -> None:
    from noelle.app.services import hf_client
    from noelle.app.ui.setup_assistant import SetupAssistant, should_run

    assert should_run(ctx.settings)
    wizard = SetupAssistant(ctx)
    qtbot.addWidget(wizard)
    wizard.show()
    assert wizard.currentId() == SetupAssistant.WELCOME
    wizard.next()
    account = wizard.currentPage()
    account.token.setText("hf_fromWizard")
    wizard.next()  # validatePage stores the token
    assert hf_client.get_token() == "hf_fromWizard"
    assert wizard.currentId() == SetupAssistant.HARDWARE
    assert "parameters" in wizard.currentPage().advice.text.text()
    wizard.next()
    downloads = wizard.currentPage()
    qtbot.waitUntil(lambda: downloads.release is not None, timeout=3000)
    assert "b1" in downloads.llama_info.text() and "15.8 MB" in downloads.llama_info.text()
    assert not downloads.llama.isChecked() and not downloads.starter.isChecked()  # opt-in only
    assert downloads.nextId() == SetupAssistant.FINISH
    wizard.next()
    assert wizard.currentId() == SetupAssistant.FINISH
    wizard.accept()
    assert ctx.extras["installs"] == []  # nothing was downloaded
    assert not ctx.downloads.jobs
    assert ctx.settings.get("setup.completed") is True and not should_run(ctx.settings)


def test_ticking_llama_cpp_routes_through_install_page(ctx: Any, qtbot: Any) -> None:
    from noelle.app.ui.setup_assistant import SetupAssistant

    wizard = SetupAssistant(ctx)
    qtbot.addWidget(wizard)
    wizard.show()
    for _ in range(3):
        wizard.next()
    downloads = wizard.currentPage()
    qtbot.waitUntil(lambda: downloads.release is not None, timeout=3000)
    downloads.llama.setChecked(True)
    assert downloads.nextId() == SetupAssistant.INSTALL
    wizard.next()
    qtbot.waitUntil(lambda: len(ctx.extras["installs"]) == 1, timeout=3000)
    qtbot.waitUntil(lambda: wizard.currentPage().isComplete(), timeout=3000)
