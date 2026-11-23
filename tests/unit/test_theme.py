"""Light (white) and dark (neutral grey) palettes, chosen at start-up from ``ui.theme``."""

from __future__ import annotations

import os
import re
from typing import Any, Iterator

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture()
def restore_theme() -> Iterator[Any]:
    from noelle.app import theme

    before = theme.NAME
    yield theme
    theme.use(before)


def test_palettes_define_the_same_tokens() -> None:
    from noelle.app import theme

    assert set(theme.LIGHT) == set(theme.DARK)
    assert set(theme.THEMES) == {"system", "light", "dark"}


def test_use_switches_every_token(restore_theme: Any) -> None:
    theme = restore_theme
    assert theme.use("light") == "light"
    assert theme.BG == "#FFFFFF" and not theme.IS_DARK
    assert theme.LEVEL_COLORS["green"] == (theme.SUCCESS, theme.SUCCESS_SOFT)
    assert theme.use("dark") == "dark"
    assert theme.IS_DARK and theme.BG == theme.DARK["BG"]
    assert theme.use("bogus") == "dark"  # unknown names fall back instead of leaving tokens empty


@pytest.mark.parametrize("name", ["light", "dark"])
def test_dark_grey_is_neutral_and_stylesheet_has_no_stray_colours(restore_theme: Any, name: str) -> None:
    theme = restore_theme
    theme.use(name)
    sheet = theme.stylesheet()
    for retired in ("#0D1015", "#151A22", "#1D2430", "#0F131A", "#333D4D", "#3D4859", "#231D45", "#0C0A1A"):
        if retired == theme.ON_ACCENT:
            continue
        assert retired not in sheet, retired
    assert "{" not in re.sub(r"\{[^{}]*\}", "", sheet)  # every f-string placeholder was filled
    if name == "dark":  # neutral grey: red, green and blue channels equal for the main surfaces
        for token in ("BG", "BG_ELEV", "SURFACE", "SURFACE_2", "SURFACE_3", "SURFACE_4", "BORDER"):
            value = theme.DARK[token].lstrip("#")
            assert value[0:2] == value[2:4] == value[4:6], token


def test_resolve_explicit_and_system(restore_theme: Any) -> None:
    theme = restore_theme
    assert theme.resolve("light") == "light"
    assert theme.resolve("dark") == "dark"
    assert theme.resolve("system") in ("light", "dark")
    assert theme.resolve(None) in ("light", "dark")


def test_apply_styles_the_application(qapp: Any, restore_theme: Any) -> None:
    from PySide6.QtGui import QPalette

    theme = restore_theme
    assert theme.apply(qapp, "light") == "light"
    assert qapp.palette().color(QPalette.ColorRole.Window).name().upper() == "#FFFFFF"
    assert "background: #FFFFFF" in qapp.styleSheet()
    assert theme.apply(qapp, "dark") == "dark"
    assert qapp.palette().color(QPalette.ColorRole.Window).name().upper() == theme.DARK["BG"]


def test_theme_setting_defaults_to_system() -> None:
    from noelle.config import get_settings

    assert get_settings().get("ui.theme") == "system"


def test_icons_follow_the_active_palette(qapp: Any, restore_theme: Any) -> None:
    from noelle.app import icons

    theme = restore_theme
    theme.use("light")
    light = icons.pixmap("check").toImage().pixelColor(18, 18)
    theme.use("dark")
    dark = icons.pixmap("check").toImage().pixelColor(18, 18)
    assert light != dark or light.alpha() == 0
