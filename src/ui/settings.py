"""Application settings and API-key resolution.

Two concerns live here:

1. ``AppSettings`` — a thin ``QSettings`` wrapper for UI state (window geometry,
   last-used portfolio, theme, update-check opt-in). Requires PySide6.

2. ``get_api_key`` — resolves optional data-provider keys (FMP, Alpha Vantage)
   without requiring Qt, so the headless pipeline / CLI can call it. Resolution
   order: process environment → ``.env`` in the config dir → ``QSettings``
   (only if PySide6 is importable). Returns ``None`` when unset — callers decide
   whether a missing key is fatal.
"""

from __future__ import annotations

import os
from typing import Optional

from . import paths

# Organization / application identifiers — must match what main_desktop.py sets
# on the QApplication so QSettings reads and writes the same store.
ORG_NAME = "PortfolioAnalyzer"
APP_NAME = "Portfolio Analyzer"

# Keys we know how to look up, mapped to their canonical env-var name.
_API_ENV_VARS = {
    "FMP_API_KEY": "FMP_API_KEY",
    "ALPHAVANTAGE_API_KEY": "ALPHAVANTAGE_API_KEY",
    "FRED_API_KEY": "FRED_API_KEY",
    # Real-time quote providers (Live Market Watch). Any one upgrades quotes from
    # yfinance-delayed to that provider's real-time feed.
    "FINNHUB_API_KEY": "FINNHUB_API_KEY",
    "POLYGON_API_KEY": "POLYGON_API_KEY",
    "ALPACA_API_KEY": "ALPACA_API_KEY",
    "ALPACA_API_SECRET": "ALPACA_API_SECRET",
}

# Real-time quote providers in priority order: (name, key settings-name,
# optional secret settings-name). The first fully-configured one wins.
REALTIME_PROVIDERS = [
    ("finnhub", "FINNHUB_API_KEY", None),
    ("polygon", "POLYGON_API_KEY", None),
    ("alpaca", "ALPACA_API_KEY", "ALPACA_API_SECRET"),
]


def _load_dotenv_value(name: str) -> Optional[str]:
    """Read a single key from the ``.env`` in the config dir, if present."""
    env_path = paths.config_dir() / ".env"
    if not env_path.exists():
        return None
    try:
        from dotenv import dotenv_values

        return dotenv_values(env_path).get(name) or None
    except Exception:
        return None


def _qsettings_value(name: str) -> Optional[str]:
    """Read a key from QSettings, but only if PySide6 is available."""
    try:
        from PySide6.QtCore import QSettings
    except Exception:
        return None
    settings = QSettings(ORG_NAME, APP_NAME)
    val = settings.value(f"api_keys/{name}")
    return str(val) if val else None


def get_api_key(name: str) -> Optional[str]:
    """Resolve an optional API key. Returns ``None`` if not configured."""
    env_name = _API_ENV_VARS.get(name, name)
    return (
        os.getenv(env_name)
        or _load_dotenv_value(env_name)
        or _qsettings_value(name)
    )


def realtime_provider():
    """Return ``(provider_name, creds)`` for the first fully-configured real-time
    quote provider, or ``(None, None)`` when none is set (yfinance-delayed).

    ``creds`` is the API key string, or ``(key, secret)`` for providers that need
    both (Alpaca). Resolution reuses :func:`get_api_key` (env / .env / QSettings).
    """
    for name, key_name, secret_name in REALTIME_PROVIDERS:
        key = get_api_key(key_name)
        if not key:
            continue
        if secret_name:
            secret = get_api_key(secret_name)
            if not secret:
                continue  # needs both key and secret
            return name, (key, secret)
        return name, key
    return None, None


class AppSettings:
    """UI-state settings backed by ``QSettings`` (PySide6 required)."""

    def __init__(self, settings=None) -> None:
        if settings is None:
            from PySide6.QtCore import QSettings

            settings = QSettings(ORG_NAME, APP_NAME)
        self._s = settings

    # ── Window geometry ──
    def save_geometry(self, window) -> None:
        self._s.setValue("window/geometry", window.saveGeometry())
        self._s.setValue("window/state", window.saveState())

    def restore_geometry(self, window) -> None:
        geo = self._s.value("window/geometry")
        state = self._s.value("window/state")
        if geo is not None:
            window.restoreGeometry(geo)
        if state is not None:
            window.restoreState(state)

    # ── Generic prefs ──
    def get(self, key: str, default=None):
        val = self._s.value(key, default)
        return val

    def set(self, key: str, value) -> None:
        self._s.setValue(key, value)

    def set_api_key(self, name: str, value: str) -> None:
        self._s.setValue(f"api_keys/{name}", value)
