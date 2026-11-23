"""Build a distributable Noelle package with PyInstaller, then optionally sign it.

    python packaging/build.py            # build for the current platform
    python packaging/build.py --sign     # also sign (needs the environment variables below)

Signing is driven by environment variables so no secret ever lives in the repository:

* macOS:   NOELLE_MAC_SIGN_IDENTITY ("Developer ID Application: ..."), and for notarisation
           NOELLE_NOTARY_PROFILE (a profile stored with `xcrun notarytool store-credentials`).
* Windows: NOELLE_WIN_CERT_PATH and NOELLE_WIN_CERT_PASSWORD (a .pfx used with signtool).
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"


def run(cmd: list[str]) -> None:
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)


def build(clean: bool) -> Path:
    cmd = [sys.executable, "-m", "PyInstaller", str(ROOT / "packaging" / "noelle.spec"), "--noconfirm",
           "--distpath", str(DIST), "--workpath", str(ROOT / "build")]
    if clean:
        cmd.append("--clean")
    run(cmd)
    return DIST / ("Noelle.app" if platform.system() == "Darwin" else "Noelle")


def sign_macos(app: Path) -> None:
    identity = os.environ.get("NOELLE_MAC_SIGN_IDENTITY")
    if not identity:
        print("NOELLE_MAC_SIGN_IDENTITY not set; skipping signing")
        return
    run(["codesign", "--deep", "--force", "--options", "runtime", "--timestamp", "--entitlements",
         str(ROOT / "packaging" / "entitlements.plist"), "--sign", identity, str(app)])
    profile = os.environ.get("NOELLE_NOTARY_PROFILE")
    if profile:
        archive = DIST / "Noelle.zip"
        run(["ditto", "-c", "-k", "--keepParent", str(app), str(archive)])
        run(["xcrun", "notarytool", "submit", str(archive), "--keychain-profile", profile, "--wait"])
        run(["xcrun", "stapler", "staple", str(app)])


def sign_windows(folder: Path) -> None:
    cert = os.environ.get("NOELLE_WIN_CERT_PATH")
    password = os.environ.get("NOELLE_WIN_CERT_PASSWORD")
    if not cert or not password:
        print("NOELLE_WIN_CERT_PATH/NOELLE_WIN_CERT_PASSWORD not set; skipping signing")
        return
    signtool = shutil.which("signtool") or "signtool"
    run([signtool, "sign", "/f", cert, "/p", password, "/fd", "sha256", "/tr", "http://timestamp.digicert.com",
         "/td", "sha256", str(folder / "Noelle.exe")])


def archive(target: Path) -> Path:
    system = platform.system().lower()
    machine = platform.machine().lower()
    base = DIST / f"Noelle-1.0.0-{system}-{machine}"
    if system == "darwin":
        out = base.with_suffix(".zip")
        run(["ditto", "-c", "-k", "--keepParent", str(target), str(out)])
        return out
    return Path(shutil.make_archive(str(base), "zip" if system == "windows" else "gztar", target.parent,
                                    target.name))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sign", action="store_true")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    target = build(args.clean)
    if args.sign:
        if platform.system() == "Darwin":
            sign_macos(target)
        elif platform.system() == "Windows":
            sign_windows(target)
    print("archive:", archive(target))
    return 0


if __name__ == "__main__":
    sys.exit(main())
