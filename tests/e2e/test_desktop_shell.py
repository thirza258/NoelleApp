"""Offscreen UI tests (pytest-qt). Hub calls are replaced with fixtures; no network needed."""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from typing import Any

import pytest

pytestmark = pytest.mark.e2e
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _models(prefix: str, count: int) -> list[Any]:
    from noelle.app.services.hf_client import ModelSummary

    return [ModelSummary(id=f"{prefix}/model-{i}-{(i % 7) + 1}B", author=prefix, pipeline_tag="text-generation",
                         downloads=1000 * i, likes=i, trending_score=100 - i, license="apache-2.0",
                         param_count=((i % 7) + 1) * 10**9, last_modified=datetime.now(timezone.utc))
            for i in range(count)]


class FakePager:
    def __init__(self, pages: int = 3, size: int = 30) -> None:
        self._pages = [_models(f"org{p}", size) for p in range(pages)]
        self.exhausted = False

    def fetch_next(self) -> list[Any]:
        if not self._pages:
            self.exhausted = True
            return []
        page = self._pages.pop(0)
        self.exhausted = not self._pages
        return page


@pytest.fixture()
def window(qtbot: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
    from noelle.app import main as appmain
    from noelle.app import theme
    from noelle.app.services.hf_client import HFClient
    from noelle.app.ui.main_window import MainWindow
    from PySide6.QtWidgets import QApplication

    monkeypatch.setattr(HFClient, "pager", lambda self, **kw: FakePager())
    monkeypatch.setattr(HFClient, "trending", lambda self, count=5, tag=None: _models("hero", count))
    monkeypatch.setattr(HFClient, "fetch_readme", lambda self, repo: "A test model card paragraph long enough to show.")
    monkeypatch.setattr(HFClient, "whoami", lambda self, token=None: {"name": "tester"})
    theme.apply(QApplication.instance())
    ctx = appmain.build_context()
    win = MainWindow(ctx)
    win.resize(1440, 900)
    qtbot.addWidget(win)
    win.show()
    ctx.hardware.start()
    win.start()
    yield win
    ctx.hardware.stop()
    win.close()


def test_navigation_and_grid_population(window: Any, qtbot: Any) -> None:
    page = window.pages["discover"]
    qtbot.waitUntil(lambda: len(page.grid.cards) >= 30, timeout=5000)
    assert page.hero.current() is not None
    for key in ("workspace", "compare", "storage", "server", "discover"):
        window.navigate(key)
        assert window.stack.currentWidget() is window.pages[key]


def test_hero_collapses_and_expands_with_scroll(window: Any, qtbot: Any) -> None:
    page = window.pages["discover"]
    qtbot.waitUntil(lambda: len(page.grid.cards) >= 30, timeout=5000)
    qtbot.waitUntil(lambda: page.grid.columns > 0, timeout=2000)
    columns_before = page.grid.columns
    assert page.hero_column.width() >= 370
    bar = page.scroll.verticalScrollBar()
    qtbot.waitUntil(lambda: bar.maximum() > 400, timeout=3000)  # wait for the scroll range to be laid out
    started = time.monotonic()
    bar.setValue(121)
    qtbot.waitUntil(lambda: not page.hero_column.isVisible(), timeout=1500)
    elapsed = time.monotonic() - started
    assert elapsed < 0.6  # 250 ms animation plus event-loop slack
    assert page.hero_column.maximumWidth() == 0
    qtbot.waitUntil(lambda: page.grid.columns > columns_before, timeout=1500)
    # hysteresis: small scroll changes away from the top must not re-expand the hero
    bar.setValue(60)
    qtbot.wait(350)
    assert not page.hero_column.isVisible()
    bar.setValue(0)
    qtbot.waitUntil(lambda: page.hero_column.isVisible() and page.hero_column.width() >= 370, timeout=1500)
    qtbot.waitUntil(lambda: page.grid.columns == columns_before, timeout=1500)


def test_infinite_scroll_loads_next_page(window: Any, qtbot: Any) -> None:
    page = window.pages["discover"]
    qtbot.waitUntil(lambda: len(page.grid.cards) >= 30, timeout=5000)
    bar = page.scroll.verticalScrollBar()
    for _ in range(5):
        bar.setValue(bar.maximum())
        qtbot.wait(200)
    qtbot.waitUntil(lambda: len(page.grid.cards) >= 60, timeout=5000)


def test_compatibility_badges_follow_budget(window: Any) -> None:
    from noelle.runtime.preflight import GB, MemoryBudget

    ctx = window.ctx
    ctx.hardware.memory_budget = lambda headroom=0.15: MemoryBudget(int(8 * GB), headroom, int(16 * GB), "mps")
    big, small = _models("x", 7)[6], _models("x", 7)[0]  # 7B and 1B
    assert ctx.compat(small).level == "green"
    assert ctx.compat(big).level in ("yellow", "red")
