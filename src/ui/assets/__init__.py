"""Static assets (logo, icons) for the desktop UI."""

from __future__ import annotations

from pathlib import Path

_ASSETS_DIR = Path(__file__).resolve().parent


def logo_path() -> str:
    """Absolute path to the full logo SVG (mark inside a rounded Graphite tile).
    Used for the window/taskbar/tray icon and packaging (app.ico / app.icns),
    where the tile keeps the mark legible on light and dark desktops."""
    return str(_ASSETS_DIR / "logo.svg")


def mark_path() -> str:
    """Absolute path to the bare logo mark (allocation ring + bars, no tile) —
    used on dark surfaces: the in-app header and the report title slide."""
    return str(_ASSETS_DIR / "logo_mark.svg")


def asset(name: str) -> str:
    """Absolute path to any file in the assets directory (e.g. an icon SVG)."""
    return str(_ASSETS_DIR / name)
