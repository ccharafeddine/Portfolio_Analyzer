"""Static assets (logo, icons) for the desktop UI."""

from __future__ import annotations

from pathlib import Path

_ASSETS_DIR = Path(__file__).resolve().parent


def logo_path() -> str:
    """Absolute path to the full logo SVG (mark inside a rounded app tile).
    Kept for packaging (.ico/.icns), where the tiled form is conventional."""
    return str(_ASSETS_DIR / "logo.svg")


def mark_path() -> str:
    """Absolute path to the bare logo mark (donut + bars, no app tile) — used for
    the window/taskbar icon and the in-app header."""
    return str(_ASSETS_DIR / "logo_mark.svg")


def asset(name: str) -> str:
    """Absolute path to any file in the assets directory (e.g. an icon SVG)."""
    return str(_ASSETS_DIR / name)
