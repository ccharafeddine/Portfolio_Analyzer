"""Tests for the auto-update version logic (no network)."""

from __future__ import annotations

from src.ui import updater


def test_parse_version():
    assert updater._parse_version("v1.5.0") == (1, 5, 0)
    assert updater._parse_version("1.5") == (1, 5, 0)
    assert updater._parse_version("v2.0.1-beta") == (2, 0, 1)
    assert updater._parse_version("") == (0, 0, 0)


def test_is_newer():
    assert updater.is_newer("v1.6.0", "1.5.0")
    assert updater.is_newer("v1.5.1", "1.5.0")
    assert updater.is_newer("2.0.0", "1.9.9")
    assert not updater.is_newer("v1.5.0", "1.5.0")
    assert not updater.is_newer("v1.4.9", "1.5.0")


def test_pick_asset_windows(monkeypatch):
    monkeypatch.setattr(updater.platform, "system", lambda: "Windows")
    assets = [
        {"name": "PortfolioAnalyzer.dmg", "browser_download_url": "u.dmg"},
        {"name": "PortfolioAnalyzer-Setup.exe", "browser_download_url": "u.exe"},
    ]
    assert updater._pick_asset(assets) == "u.exe"


def test_pick_asset_macos(monkeypatch):
    monkeypatch.setattr(updater.platform, "system", lambda: "Darwin")
    assets = [
        {"name": "PortfolioAnalyzer-Setup.exe", "browser_download_url": "u.exe"},
        {"name": "PortfolioAnalyzer.dmg", "browser_download_url": "u.dmg"},
    ]
    assert updater._pick_asset(assets) == "u.dmg"


def test_pick_asset_none_when_missing(monkeypatch):
    monkeypatch.setattr(updater.platform, "system", lambda: "Windows")
    assert updater._pick_asset([{"name": "notes.txt", "browser_download_url": "u"}]) is None
