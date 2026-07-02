"""Cross-platform filesystem locations for the desktop app.

Single source of truth for where the app reads and writes files, so the same
logic works on Windows and macOS without hard-coded paths. Uses ``platformdirs``
for OS-correct locations:

- machine state (cache, per-run outputs, config, saved portfolios) lives under
  the per-user *data dir* (e.g. ``%LOCALAPPDATA%\\Portfolio Analyzer`` on
  Windows, ``~/Library/Application Support/Portfolio Analyzer`` on macOS);
- artifacts the user explicitly exports default to their ``Documents`` folder.

Every getter creates the directory on demand so callers never have to.
"""

from __future__ import annotations

from pathlib import Path

from platformdirs import PlatformDirs

# Keep author/appname stable — changing these moves every user's data.
_APP_NAME = "Portfolio Analyzer"
_APP_AUTHOR = "PortfolioAnalyzer"

_dirs = PlatformDirs(appname=_APP_NAME, appauthor=_APP_AUTHOR)


def _ensure(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def data_dir() -> Path:
    """Root for all per-user application state."""
    return _ensure(Path(_dirs.user_data_dir))


def config_dir() -> Path:
    """Holds app settings and the ``.env`` for API keys."""
    return _ensure(Path(_dirs.user_config_dir))


def cache_dir() -> Path:
    """Parquet price cache. Replaces the fetcher's ``~/.portfolio_analyzer_cache``."""
    return _ensure(Path(_dirs.user_cache_dir) / "prices")


def portfolios_dir() -> Path:
    """Saved ``PortfolioConfig`` JSON files (multi-portfolio store)."""
    return _ensure(data_dir() / "portfolios")


def outputs_dir() -> Path:
    """Per-run pipeline outputs (CSVs, generated reports)."""
    return _ensure(data_dir() / "outputs")


def documents_export_dir() -> Path:
    """Default target for user-initiated exports (reports, ZIPs)."""
    return _ensure(Path(_dirs.user_documents_dir) / _APP_NAME)
