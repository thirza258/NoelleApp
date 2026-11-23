"""Render the app icon into installer formats: noelle.png (512), noelle.ico (Windows), noelle.icns (macOS).

    python packaging/make_icons.py

The .icns step uses macOS `iconutil`; on other platforms only the PNG and ICO are written.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))


def main() -> int:
    from PIL import Image
    from PySide6.QtCore import QSize
    from PySide6.QtWidgets import QApplication

    from noelle.app import icons

    QApplication([])  # QPixmap needs an application instance
    icon = icons.app_icon()

    def render(size: int, target: Path) -> None:
        icon.pixmap(QSize(size, size)).toImage().scaled(size, size).save(str(target), "PNG")

    render(512, HERE / "noelle.png")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        sizes = [16, 24, 32, 48, 64, 128, 256]
        for size in sizes:
            render(size, tmp_dir / f"{size}.png")
        base = Image.open(tmp_dir / "256.png")
        base.save(HERE / "noelle.ico", sizes=[(s, s) for s in sizes])
        if shutil.which("iconutil"):
            iconset = tmp_dir / "noelle.iconset"
            iconset.mkdir()
            for size in (16, 32, 128, 256, 512):
                render(size, iconset / f"icon_{size}x{size}.png")
                render(size * 2, iconset / f"icon_{size}x{size}@2x.png")
            subprocess.run(["iconutil", "-c", "icns", str(iconset), "-o", str(HERE / "noelle.icns")], check=True)
    print("wrote", ", ".join(p.name for p in sorted(HERE.glob("noelle.*")) if p.suffix in (".png", ".ico", ".icns")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
