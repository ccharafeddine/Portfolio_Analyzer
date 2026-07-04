"""Live Market Watch dashboard: splitter + panel-visibility persistence.

Qt-dependent, so it is skipped wherever PySide6.QtWidgets can't be imported or a
(headless) QApplication can't be created — matching the project's "keep the
default test run Qt-free" stance while still exercising the real widgets when Qt
is present (locally / on CI runners that set ``QT_QPA_PLATFORM=offscreen``).

Layout state round-trips through a real ``AppSettings`` backed by a throwaway
QSettings ini file, so nothing touches the developer's live settings store.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

# Import Qt behind a broad guard and skip the whole module if it can't load —
# the CI Ubuntu runner has no libEGL, so ``import PySide6.QtWidgets`` raises a
# plain ImportError (not ModuleNotFoundError, so ``importorskip`` would re-raise
# it). A QApplication must also exist before any QWidget is constructed.
try:
    from PySide6 import QtCore, QtWidgets

    _APP = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
except Exception as exc:  # pragma: no cover - headless runner without libEGL
    pytest.skip(f"Qt unavailable: {exc}", allow_module_level=True)

from src.ui.live_watch_view import (  # noqa: E402  (must follow the QApplication guard)
    PANEL_DAYCHANGE,
    PANEL_PRICE,
    PANEL_TREEMAP,
    PANEL_WATCHLIST,
    LiveWatchView,
)
from src.ui.settings import AppSettings  # noqa: E402


def _settings(tmp_path):
    """A real AppSettings over an isolated ini file (round-trips QByteArray)."""
    qs = QtCore.QSettings(str(tmp_path / "market_watch.ini"), QtCore.QSettings.IniFormat)
    return AppSettings(qs)


def _build(settings):
    view = LiveWatchView(settings=settings)
    view.resize(1000, 700)
    return view


def test_panel_visibility_round_trips(tmp_path):
    settings = _settings(tmp_path)

    v1 = _build(settings)
    # Toggle a distinctive mixed pattern (some off, some on).
    v1.set_panel_visible(PANEL_PRICE, False)
    v1.set_panel_visible(PANEL_TREEMAP, True)
    v1.set_panel_visible(PANEL_DAYCHANGE, False)
    v1.set_panel_visible(PANEL_WATCHLIST, False)
    v1.shutdown()  # persists

    # A freshly built view over the same settings restores the exact pattern.
    v2 = _build(settings)
    assert v2.panel_visible(PANEL_PRICE) is False
    assert v2.panel_visible(PANEL_TREEMAP) is True
    assert v2.panel_visible(PANEL_DAYCHANGE) is False
    assert v2.panel_visible(PANEL_WATCHLIST) is False

    # ...and it actually applied to the widgets, not just the tracking dict.
    assert v2._chart._price_box.isHidden() is True
    assert v2._chart._daychange_box.isHidden() is True
    assert v2._chart._treemap.isHidden() is False
    assert v2._tabs.isTabVisible(v2._watchlist_tab_index) is False
    # The menu actions reflect the restored state too.
    assert v2._panel_actions[PANEL_PRICE].isChecked() is False
    assert v2._panel_actions[PANEL_TREEMAP].isChecked() is True
    v2.shutdown()


def test_splitter_geometry_round_trips(tmp_path):
    settings = _settings(tmp_path)

    v1 = _build(settings)
    v1._h_splitter.setSizes([250, 750])
    v1._chart.splitter().setSizes([300, 200, 200])
    v1._save_layout()
    saved_h = bytes(v1._h_splitter.saveState())
    saved_v = bytes(v1._chart.splitter().saveState())
    v1.shutdown()

    v2 = _build(settings)
    # restoreState -> saveState reproduces identical bytes for the same child
    # count, proving the geometry was restored from settings.
    assert bytes(v2._h_splitter.saveState()) == saved_h
    assert bytes(v2._chart.splitter().saveState()) == saved_v
    v2.shutdown()


def test_smoke_toggle_each_panel_off_then_on(tmp_path):
    """Build offscreen, toggle every panel off then on, resize a splitter, tear
    down and rebuild, and confirm visibility + sizes restore from settings."""
    settings = _settings(tmp_path)

    v1 = _build(settings)
    keys = [PANEL_PRICE, PANEL_TREEMAP, PANEL_DAYCHANGE, PANEL_WATCHLIST]
    for k in keys:  # off
        v1.set_panel_visible(k, False)
        assert v1.panel_visible(k) is False
    for k in keys:  # back on
        v1.set_panel_visible(k, True)
        assert v1.panel_visible(k) is True
    # Leave one off to make the restored state non-trivial, and resize a splitter.
    v1.set_panel_visible(PANEL_DAYCHANGE, False)
    v1._h_splitter.setSizes([400, 600])
    v1._save_layout()
    saved_h = bytes(v1._h_splitter.saveState())
    v1.shutdown()

    v2 = _build(settings)
    assert v2.panel_visible(PANEL_DAYCHANGE) is False
    assert v2.panel_visible(PANEL_PRICE) is True
    assert v2.panel_visible(PANEL_WATCHLIST) is True
    assert bytes(v2._h_splitter.saveState()) == saved_h
    v2.shutdown()
