# PyInstaller spec for Portfolio Analyzer.
#
# Produces a one-directory build (recommended for QtWebEngine) — a Windows
# folder with PortfolioAnalyzer.exe, and a macOS .app bundle. Build with:
#
#     pyinstaller packaging/PortfolioAnalyzer.spec --noconfirm
#
# The release workflow zips the resulting dist/ output per OS.

import os
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_all

block_cipher = None

# Repo root, resolved from the spec's own location so the build works regardless
# of the current working directory. ``SPECPATH`` is injected by PyInstaller.
ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))

datas = []
binaries = []
hiddenimports = []

# Bundle package data + dynamic pieces PyInstaller can miss:
#  - plotly: get_plotlyjs() reads the bundled plotly.min.js at runtime
#  - kaleido: static chart export for PDF/HTML/PowerPoint reports
#  - statsmodels / scipy: data tables and compiled bits
#  - keyring: backends (Windows Credential Locker / macOS Keychain) are resolved
#    via entry points that PyInstaller misses without collect_all
for pkg in ("plotly", "kaleido", "statsmodels", "keyring"):
    try:
        d, b, h = collect_all(pkg)
        datas += d
        binaries += b
        hiddenimports += h
    except Exception:
        pass

# App assets (logos, chevrons, app.ico). Kept under the same relative path so the
# code's ``Path(__file__).parent`` asset resolution works when frozen.
datas += [(os.path.join(ROOT, "src/ui/assets"), "src/ui/assets")]

ICON_ICO = Path(ROOT) / "src/ui/assets/app.ico"
ICON_ICNS = Path(ROOT) / "src/ui/assets/app.icns"
if sys.platform == "win32":
    icon = str(ICON_ICO) if ICON_ICO.exists() else None
elif sys.platform == "darwin":
    icon = str(ICON_ICNS) if ICON_ICNS.exists() else None
else:
    icon = None

a = Analysis(
    [os.path.join(ROOT, "main_desktop.py")],
    pathex=[ROOT],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=["streamlit", "tkinter"],
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="PortfolioAnalyzer",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    icon=icon,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="PortfolioAnalyzer",
)

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="Portfolio Analyzer.app",
        icon=icon,
        bundle_identifier="com.ccharafeddine.portfolioanalyzer",
        info_plist={
            "CFBundleName": "Portfolio Analyzer",
            "CFBundleDisplayName": "Portfolio Analyzer",
            "NSHighResolutionCapable": True,
        },
    )
