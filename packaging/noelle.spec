# PyInstaller spec for Noelle Desktop AI Studio (task T-703).
#
# Build:   python packaging/build.py            (wraps: pyinstaller packaging/noelle.spec)
# Output:  dist/Noelle/            one-folder build (Windows/Linux)
#          dist/Noelle.app         macOS bundle
#
# One-folder (not one-file) keeps start-up fast: torch alone is several hundred MB and a
# one-file build would unpack it to a temp dir on every launch. The same executable serves
# as desktop shell, inference worker, API server and download child (see noelle/__main__.py).
# ruff: noqa
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata

ROOT = Path(SPECPATH).parent
APP_NAME = "Noelle"
IS_MAC = sys.platform == "darwin"

hiddenimports = (
    collect_submodules("noelle")
    + collect_submodules("zmq")
    + collect_submodules("uvicorn")
    + ["keyring.backends.macOS", "keyring.backends.Windows", "keyring.backends.SecretService",
       "sentence_transformers", "pypdf", "docx", "sqlite_vec", "peft", "diffusers"]
)
if sys.platform == "win32":
    hiddenimports += collect_submodules("win32ctypes")  # keyring's Windows Credential Manager client
if IS_MAC:
    hiddenimports += collect_submodules("mlx_lm")

datas = collect_data_files("noelle")
for package in ("transformers", "diffusers", "tokenizers", "sqlite_vec", "huggingface_hub"):
    try:
        datas += collect_data_files(package)
    except Exception:
        pass
if IS_MAC:
    datas += collect_data_files("mlx")
# packages that read their own version metadata at import time
for package in ("transformers", "huggingface_hub", "tokenizers", "safetensors", "accelerate", "torch", "tqdm",
                "regex", "requests", "packaging", "filelock", "numpy", "pyyaml", "diffusers", "peft",
                "sentence-transformers", "keyring"):
    try:
        datas += copy_metadata(package)
    except Exception:
        pass

a = Analysis(
    [str(ROOT / "packaging" / "launcher.py")],
    pathex=[str(ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=["tkinter", "matplotlib", "IPython.kernel", "pytest", "pytest_qt"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=APP_NAME,
    debug=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=True,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=str(ROOT / "packaging" / "entitlements.plist") if IS_MAC else None,
    icon=str(ROOT / "packaging" / ("noelle.icns" if IS_MAC else "noelle.ico"))
    if (ROOT / "packaging" / ("noelle.icns" if IS_MAC else "noelle.ico")).exists() else None,
)

coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name=APP_NAME)

if IS_MAC:
    app = BUNDLE(
        coll,
        name=f"{APP_NAME}.app",
        bundle_identifier="ai.noelle.studio",
        icon=str(ROOT / "packaging" / "noelle.icns"),
        info_plist={
            "CFBundleName": APP_NAME,
            "CFBundleDisplayName": "Noelle AI Studio",
            "CFBundleShortVersionString": "1.3.4",
            "CFBundleVersion": "1.3.4",
            "LSApplicationCategoryType": "public.app-category.developer-tools",
            "NSHighResolutionCapable": True,
            "LSMinimumSystemVersion": "13.0",
            "NSMicrophoneUsageDescription": "Noelle records audio locally for speech recognition. "
                                            "Recordings never leave your Mac.",
            "NSCameraUsageDescription": "Noelle opens the camera only when you choose Camera in the vision "
                                        "playground. Frames are processed locally and never uploaded.",
        },
    )
