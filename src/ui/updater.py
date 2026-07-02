"""Auto-update: check the GitHub Releases API for a newer version.

Stdlib-only (urllib/json). The check runs on a background QThread so the UI never
blocks; the main window shows the result (up-to-date / update-available / error).
"""

from __future__ import annotations

import json
import platform
import re
import urllib.request
from typing import Optional

from PySide6.QtCore import QObject, Signal, Slot

GITHUB_REPO = "ccharafeddine/Portfolio_Analyzer"
_LATEST_URL = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"
_RELEASES_PAGE = f"https://github.com/{GITHUB_REPO}/releases/latest"


def _parse_version(tag: str) -> tuple:
    """'v1.5.0' -> (1, 5, 0). Non-numeric parts are ignored; missing -> 0."""
    nums = re.findall(r"\d+", tag or "")
    return tuple(int(n) for n in nums[:3]) + (0,) * (3 - len(nums[:3]))


def is_newer(latest: str, current: str) -> bool:
    """True if ``latest`` is a strictly higher version than ``current``."""
    return _parse_version(latest) > _parse_version(current)


def _pick_asset(assets: list[dict]) -> Optional[str]:
    """Best download URL for this OS (.exe on Windows, .dmg on macOS)."""
    want = ".exe" if platform.system() == "Windows" else (
        ".dmg" if platform.system() == "Darwin" else None
    )
    if want:
        for a in assets:
            name = (a.get("name") or "").lower()
            if name.endswith(want):
                return a.get("browser_download_url")
    return None


def fetch_latest_release(timeout: float = 12.0) -> dict:
    """Return {tag, name, notes, url, download_url}. Raises on network/parse error."""
    req = urllib.request.Request(
        _LATEST_URL,
        headers={"User-Agent": "PortfolioAnalyzer", "Accept": "application/vnd.github+json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    tag = data.get("tag_name", "")
    return {
        "tag": tag,
        "name": data.get("name") or tag,
        "notes": data.get("body") or "",
        "url": data.get("html_url") or _RELEASES_PAGE,
        "download_url": _pick_asset(data.get("assets") or []),
    }


class UpdateCheckWorker(QObject):
    """Fetches the latest release off the UI thread."""

    done = Signal(object)   # the release dict
    failed = Signal(str)

    @Slot()
    def run(self) -> None:
        try:
            self.done.emit(fetch_latest_release())
        except Exception as e:  # network, JSON, etc.
            self.failed.emit(str(e))
