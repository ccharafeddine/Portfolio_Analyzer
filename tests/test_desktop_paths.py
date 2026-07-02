"""Tests for the desktop plumbing (src/ui/paths.py, settings.py) and the
backward-compatibility of the fetcher's cache-dir override.

These import only lightweight modules — no PySide6 GUI is constructed — so they
run in the normal headless CI environment.
"""

from __future__ import annotations

import importlib
from pathlib import Path

from src.ui import paths, settings


def test_path_getters_return_existing_dirs():
    for getter in (
        paths.data_dir,
        paths.config_dir,
        paths.cache_dir,
        paths.portfolios_dir,
        paths.outputs_dir,
    ):
        p = getter()
        assert isinstance(p, Path)
        assert p.exists() and p.is_dir()


def test_cache_dir_under_data_root():
    # Cache is a distinct subtree from outputs so clearing one never nukes the other.
    assert paths.cache_dir().name == "prices"
    assert paths.outputs_dir().name == "outputs"


def test_get_api_key_reads_env(monkeypatch):
    monkeypatch.setenv("FMP_API_KEY", "test-key-123")
    assert settings.get_api_key("FMP_API_KEY") == "test-key-123"


def test_get_api_key_none_when_unset(monkeypatch):
    monkeypatch.delenv("FMP_API_KEY", raising=False)
    # Isolate from any real QSettings value on the dev machine so the test is
    # hermetic (a locally-saved key must not make this fail).
    monkeypatch.setattr(settings, "_qsettings_value", lambda name: None)
    assert settings.get_api_key("FMP_API_KEY") in (None, "")


def test_fetcher_cache_default_unchanged_without_override(monkeypatch):
    """With no env override, the fetcher keeps its original home-dir default."""
    monkeypatch.delenv("PORTFOLIO_ANALYZER_CACHE_DIR", raising=False)
    from src.data import fetcher

    fetcher = importlib.reload(fetcher)
    assert fetcher.DEFAULT_CACHE_DIR == Path.home() / ".portfolio_analyzer_cache"


def test_fetcher_cache_honors_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("PORTFOLIO_ANALYZER_CACHE_DIR", str(tmp_path / "mycache"))
    from src.data import fetcher

    fetcher = importlib.reload(fetcher)
    assert fetcher.DEFAULT_CACHE_DIR == tmp_path / "mycache"

    # Restore the module to its unpatched state for other tests.
    monkeypatch.delenv("PORTFOLIO_ANALYZER_CACHE_DIR", raising=False)
    importlib.reload(fetcher)
