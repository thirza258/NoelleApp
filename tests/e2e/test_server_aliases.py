"""Server page alias editor and gateway snippets (offscreen, no runtime, no network)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.e2e
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture()
def page(qapp: Any, qtbot: Any, tmp_path: Path) -> Any:
    from noelle.app import main as appmain
    from noelle.app.services.model_registry import ModelRecord
    from noelle.app.ui.pages.server_page import ServerPage

    ctx = appmain.build_context()
    folder = tmp_path / "snapshot"
    folder.mkdir()
    for record in (ModelRecord(repo_id="fake/chat", pipeline_tag="text-generation", local_path=str(folder)),
                   ModelRecord(repo_id="fake/chat", pipeline_tag="text-generation", source="local",
                               local_path=str(folder)),  # same api name: listed once
                   ModelRecord(repo_id="fake/sd", pipeline_tag="text-to-image", local_path=str(folder)),
                   ModelRecord(repo_id="fake/gone", pipeline_tag="text-generation",
                               local_path=str(tmp_path / "deleted"))):
        ctx.registry.upsert_model(record)
    widget = ServerPage(ctx)
    qtbot.addWidget(widget)
    widget.show()
    widget.on_activated()
    return widget


def _targets(page: Any) -> list[str]:
    return [page.alias_target.itemData(i) for i in range(page.alias_target.count())]


def _rows(page: Any) -> list[tuple[str, str]]:
    table = page.alias_table
    return [(table.item(r, 0).text(), table.item(r, 1).text()) for r in range(table.rowCount())]


def _save(page: Any, name: str, target: str | None) -> None:
    page.alias_name.setText(name)
    page.alias_target.setCurrentIndex(page.alias_target.findData(target) if target else -1)
    page.alias_save.click()


def _on_disk(page: Any) -> Any:
    """What the API server process reads: the settings file itself."""
    return json.loads(Path(page.ctx.settings.path).read_text(encoding="utf-8"))["server"]["aliases"]


def test_alias_editor_add_edit_rename_remove(page: Any) -> None:
    assert sorted(_targets(page)) == ["fake/chat", "fake/sd"]  # downloaded models only, each api name once
    for name, target, message in (("Bad Name", "fake/chat", "lower-case"), ("auto", "fake/chat", "reserved"),
                                  ("-dash", "fake/chat", "lower-case"), ("x" * 65, "fake/chat", "lower-case"),
                                  ("noelle-chat", None, "Choose the model")):
        _save(page, name, target)
        assert not page.alias_error.isHidden() and message in page.alias_error.text()
        assert page.ctx.settings.get("server.aliases") == {}

    _save(page, "noelle-chat", "fake/chat")
    assert page.alias_error.isHidden()
    assert page.ctx.settings.get("server.aliases") == {"noelle-chat": "fake/chat"}
    assert _on_disk(page) == {"noelle-chat": "fake/chat"}
    assert _rows(page) == [("noelle-chat", "fake/chat")]
    assert page.alias_remove.isEnabled()  # the saved alias stays selected for editing

    page.alias_target.setCurrentIndex(page.alias_target.findData("fake/sd"))
    page.alias_save.click()
    assert _on_disk(page) == {"noelle-chat": "fake/sd"}

    page.alias_name.setText("noelle-main")  # saving the selected alias under a new name renames it
    page.alias_save.click()
    assert _on_disk(page) == {"noelle-main": "fake/sd"}

    page.alias_new.click()
    assert page.alias_name.text() == "" and page.alias_target.currentIndex() == -1
    _save(page, "noelle-main", "fake/chat")
    assert "already exists" in page.alias_error.text()
    _save(page, "noelle-image", "fake/sd")
    assert _on_disk(page) == {"noelle-main": "fake/sd", "noelle-image": "fake/sd"}

    page.alias_table.selectRow(1)  # rows are sorted: noelle-image, noelle-main
    assert page.alias_name.text() == "noelle-main"
    assert page.alias_target.currentData() == "fake/sd"
    page.alias_remove.click()
    assert _on_disk(page) == {"noelle-image": "fake/sd"}
    assert _rows(page) == [("noelle-image", "fake/sd")]
    assert not page.alias_remove.isEnabled()


def test_alias_to_a_missing_model_is_flagged(page: Any) -> None:
    page.ctx.settings.set("server.aliases", {"old": "fake/gone"})
    page.on_activated()
    assert _rows(page) == [("old", "fake/gone  (not downloaded)")]
    page.alias_table.selectRow(0)
    assert page.alias_name.text() == "old" and page.alias_target.currentIndex() == -1


def test_gateway_snippets_never_show_the_key(page: Any) -> None:
    page.api_key.setText("nl-very-secret-key")
    page._render_snippets()
    for name in ("Images", "Knowledge"):
        text = page.snippet_editors[name].toPlainText()
        assert "nl-very-secret-key" not in text
        assert 'Authorization: Bearer $NOELLE_API_KEY' in text
    images = page.snippet_editors["Images"].toPlainText()
    assert "/v1/images/generations" in images and '"model": "fake/sd"' in images
    knowledge = page.snippet_editors["Knowledge"].toPlainText()
    assert "/v1/knowledge/query" in knowledge and '"top_k": 4' in knowledge
    page.api_key.setText("")
    page._render_snippets()
    assert "Authorization" not in page.snippet_editors["Knowledge"].toPlainText()


def test_alias_rule_matches_the_server() -> None:
    from noelle.app.ui.pages import server_page
    from noelle.runtime import server_service

    assert server_page.ALIAS_PATTERN == server_service.ALIAS_PATTERN
