"""Build Noelle and its platform installer.

    python packaging/build.py                 # app + installer for this OS
    python packaging/build.py --skip-build    # reuse dist/ from a previous PyInstaller run
    python packaging/build.py --sign          # also sign (environment variables below)

Outputs in dist/:
  macOS    Noelle-<v>-macos-<arch>.pkg  (step-by-step Installer.app wizard; opens the Setup Assistant)
           Noelle-<v>-macos-<arch>.dmg  (drag-to-Applications alternative)
  Windows  Noelle-<v>-windows-x64-setup.exe  (Inno Setup wizard with an optional Hugging Face token page)
  Linux    noelle_<v>_<arch>.deb  and  Noelle-<v>-linux-<arch>.tar.gz

Signing reads secrets from the environment only:
  macOS    NOELLE_MAC_SIGN_IDENTITY ("Developer ID Application: …") for the app,
           NOELLE_MAC_INSTALLER_IDENTITY ("Developer ID Installer: …") for the .pkg,
           NOELLE_NOTARY_PROFILE (a `xcrun notarytool store-credentials` profile) for notarisation.
  Windows  NOELLE_WIN_CERT_PATH + NOELLE_WIN_CERT_PASSWORD (.pfx for signtool).
"""

from __future__ import annotations

import argparse
import os
import platform
import plistlib
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGING = ROOT / "packaging"
DIST = ROOT / "dist"
BUILD = ROOT / "build"
sys.path.insert(0, str(ROOT))
from noelle import __version__ as VERSION  # noqa: E402


def run(cmd: list[str], **kwargs: object) -> None:
    print("$", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, cwd=ROOT, **kwargs)


def arch_name() -> str:
    machine = platform.machine().lower()
    return {"x86_64": "x64", "amd64": "x64", "aarch64": "arm64", "arm64": "arm64"}.get(machine, machine)


def build_app(clean: bool) -> None:
    cmd = [sys.executable, "-m", "PyInstaller", str(PACKAGING / "noelle.spec"), "--noconfirm",
           "--distpath", str(DIST), "--workpath", str(BUILD / "pyinstaller")]
    if clean:
        cmd.append("--clean")
    run(cmd)


def _copy_tree(src: Path, dst: Path) -> None:
    """APFS clone when possible (instant), otherwise a regular copy that keeps symlinks."""
    if sys.platform == "darwin":
        result = subprocess.run(["cp", "-cR", str(src), str(dst)])
        if result.returncode == 0:
            return
        run(["ditto", str(src), str(dst)])
        return
    shutil.copytree(src, dst, symlinks=True)


# ------------------------------------------------------------------- macOS
def sign_macos_app(app: Path) -> None:
    identity = os.environ.get("NOELLE_MAC_SIGN_IDENTITY")
    if not identity:
        print("NOELLE_MAC_SIGN_IDENTITY not set: the app stays ad-hoc signed")
        return
    run(["codesign", "--deep", "--force", "--options", "runtime", "--timestamp", "--entitlements",
         PACKAGING / "entitlements.plist", "--sign", identity, app])


def build_macos_installers(sign: bool) -> list[Path]:
    app = DIST / "Noelle.app"
    if not app.is_dir():
        raise SystemExit(f"{app} not found; build the app first")
    if sign:
        sign_macos_app(app)
    arch = arch_name()
    work = BUILD / "macos-installer"
    shutil.rmtree(work, ignore_errors=True)
    root = work / "root"
    root.mkdir(parents=True)
    _copy_tree(app, root / "Noelle.app")

    component_plist = work / "component.plist"
    run(["pkgbuild", "--analyze", "--root", root, component_plist])
    with component_plist.open("rb") as fh:
        components = plistlib.load(fh)
    for entry in components:
        entry["BundleIsRelocatable"] = False  # always install where the user chose, never "update" a copy elsewhere
    with component_plist.open("wb") as fh:
        plistlib.dump(components, fh)
    scripts = work / "scripts"
    shutil.copytree(PACKAGING / "macos" / "scripts", scripts)
    (scripts / "postinstall").chmod(0o755)
    run(["pkgbuild", "--root", root, "--component-plist", component_plist, "--identifier", "ai.noelle.studio",
         "--version", VERSION, "--install-location", "/Applications", "--scripts", scripts,
         work / "Noelle-component.pkg"])

    resources = work / "resources"
    shutil.copytree(PACKAGING / "macos" / "resources", resources)
    shutil.copyfile(ROOT / "LICENSE", resources / "LICENSE.txt")
    distribution = (PACKAGING / "macos" / "distribution.xml").read_text(encoding="utf-8")
    distribution = distribution.replace("__VERSION__", VERSION).replace(
        "__ARCH__", "arm64" if arch == "arm64" else "x86_64")
    (work / "distribution.xml").write_text(distribution, encoding="utf-8")
    pkg = DIST / f"Noelle-{VERSION}-macos-{arch}.pkg"
    unsigned = work / pkg.name
    run(["productbuild", "--distribution", work / "distribution.xml", "--resources", resources,
         "--package-path", work, unsigned])
    installer_identity = os.environ.get("NOELLE_MAC_INSTALLER_IDENTITY") if sign else None
    if installer_identity:
        run(["productsign", "--sign", installer_identity, unsigned, pkg])
    else:
        shutil.move(str(unsigned), pkg)

    dmg_root = work / "dmg"
    dmg_root.mkdir()
    _copy_tree(app, dmg_root / "Noelle.app")
    os.symlink("/Applications", dmg_root / "Applications")
    dmg = DIST / f"Noelle-{VERSION}-macos-{arch}.dmg"
    dmg.unlink(missing_ok=True)
    run(["hdiutil", "create", "-volname", "Noelle", "-srcfolder", dmg_root, "-fs", "HFS+", "-format", "UDZO",
         "-ov", dmg])

    profile = os.environ.get("NOELLE_NOTARY_PROFILE") if sign else None
    if profile and installer_identity:
        for artifact in (pkg, dmg):
            run(["xcrun", "notarytool", "submit", artifact, "--keychain-profile", profile, "--wait"])
            run(["xcrun", "stapler", "staple", artifact])
    shutil.rmtree(work, ignore_errors=True)
    return [pkg, dmg]


# ----------------------------------------------------------------- Windows
def build_windows_installer(sign: bool) -> list[Path]:
    folder = DIST / "Noelle"
    iscc = shutil.which("iscc") or r"C:\Program Files (x86)\Inno Setup 6\ISCC.exe"
    if not Path(iscc).exists() and not shutil.which("iscc"):
        raise SystemExit("Inno Setup (iscc) not found: install it from https://jrsoftware.org/isinfo.php")
    run([iscc, f"/DAppVersion={VERSION}", f"/DSourceDir={folder}", f"/DOutputDir={DIST}",
         PACKAGING / "windows" / "noelle.iss"])
    setup = DIST / f"Noelle-{VERSION}-windows-x64-setup.exe"
    cert, password = os.environ.get("NOELLE_WIN_CERT_PATH"), os.environ.get("NOELLE_WIN_CERT_PASSWORD")
    if sign and cert and password:
        signtool = shutil.which("signtool") or "signtool"
        for target in (folder / "Noelle.exe", setup):
            run([signtool, "sign", "/f", cert, "/p", password, "/fd", "sha256", "/tr",
                 "http://timestamp.digicert.com", "/td", "sha256", target])
    return [setup]


# ------------------------------------------------------------------- Linux
DEB_CONTROL = """Package: noelle
Version: {version}
Section: science
Priority: optional
Architecture: {arch}
Installed-Size: {size_kb}
Maintainer: Noelle maintainers <noreply@users.noreply.github.com>
Depends: libegl1, libgl1, libxkbcommon-x11-0, libfontconfig1, libdbus-1-3, libxcb-cursor0
Description: Local-first desktop studio for open-weight AI models
 Discover models on the Hugging Face Hub, check hardware fit, download with
 checksum verification, run chat, vision, speech, image and embedding models
 locally, compare them side by side and serve them through a local
 OpenAI-compatible API.
"""

DEB_POSTINST = """#!/bin/sh
set -e
command -v update-desktop-database >/dev/null 2>&1 && update-desktop-database -q /usr/share/applications || true
command -v gtk-update-icon-cache >/dev/null 2>&1 && gtk-update-icon-cache -q /usr/share/icons/hicolor || true
exit 0
"""


def build_linux_installers() -> list[Path]:
    folder = DIST / "Noelle"
    arch = "amd64" if arch_name() == "x64" else arch_name()
    staging = BUILD / "deb" / f"noelle_{VERSION}_{arch}"
    shutil.rmtree(staging.parent, ignore_errors=True)
    shutil.copytree(folder, staging / "opt" / "noelle", symlinks=True)
    (staging / "usr" / "bin").mkdir(parents=True)
    os.symlink("/opt/noelle/Noelle", staging / "usr" / "bin" / "noelle")
    apps = staging / "usr" / "share" / "applications"
    apps.mkdir(parents=True)
    shutil.copyfile(PACKAGING / "linux" / "noelle.desktop", apps / "noelle.desktop")
    icons = staging / "usr" / "share" / "icons" / "hicolor" / "512x512" / "apps"
    icons.mkdir(parents=True)
    shutil.copyfile(PACKAGING / "noelle.png", icons / "noelle.png")
    size_kb = sum(p.stat().st_size for p in staging.rglob("*") if p.is_file() and not p.is_symlink()) // 1024
    debian = staging / "DEBIAN"
    debian.mkdir()
    (debian / "control").write_text(DEB_CONTROL.format(version=VERSION, arch=arch, size_kb=size_kb), encoding="utf-8")
    (debian / "postinst").write_text(DEB_POSTINST, encoding="utf-8")
    (debian / "postinst").chmod(0o755)
    deb = DIST / f"noelle_{VERSION}_{arch}.deb"
    run(["dpkg-deb", "--build", "--root-owner-group", staging, deb])
    tarball = Path(shutil.make_archive(str(DIST / f"Noelle-{VERSION}-linux-{arch_name()}"), "gztar", DIST, "Noelle"))
    return [deb, tarball]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sign", action="store_true")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--skip-build", action="store_true", help="reuse the existing PyInstaller output in dist/")
    parser.add_argument("--no-installer", action="store_true")
    args = parser.parse_args()
    if not args.skip_build:
        build_app(args.clean)
    if args.no_installer:
        return 0
    system = platform.system()
    if system == "Darwin":
        outputs = build_macos_installers(args.sign)
    elif system == "Windows":
        outputs = build_windows_installer(args.sign)
    else:
        outputs = build_linux_installers()
    for path in outputs:
        print(f"installer: {path} ({path.stat().st_size / 1e6:.0f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
