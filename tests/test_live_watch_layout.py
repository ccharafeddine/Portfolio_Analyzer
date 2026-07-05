"""Live Market Watch grid dashboard: panel visibility + grid-cell persistence.

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

from src.ui.live_watch_view import LiveWatchView  # noqa: E402
from src.ui.settings import AppSettings  # noqa: E402
from src.ui.widgets.chart_heatmap_panel import (  # noqa: E402
    PANEL_CHART,
    PANEL_HEATMAP,
    PANEL_NEWS,
)
from src.ui.widgets.grid_layout import resolve  # noqa: E402


def _settings(tmp_path):
    """A real AppSettings over an isolated ini file."""
    qs = QtCore.QSettings(str(tmp_path / "market_watch.ini"), QtCore.QSettings.IniFormat)
    return AppSettings(qs)


def _build(settings):
    view = LiveWatchView(settings=settings)
    view.resize(1200, 800)
    return view


def test_panel_visibility_round_trips(tmp_path):
    settings = _settings(tmp_path)

    v1 = _build(settings)
    v1._chart.set_panel_visible(PANEL_CHART, False)   # grid card → persists via grid layout
    v1._chart.set_panel_visible(PANEL_HEATMAP, False)
    v1._set_watchlist_tab_visible(False)             # tab → persists via boolean
    v1.shutdown()

    v2 = _build(settings)
    assert v2._chart.panel_visible(PANEL_CHART) is False
    assert v2._chart.panel_visible(PANEL_HEATMAP) is False
    assert v2._chart.panel_visible(PANEL_NEWS) is True
    # The hidden grid cards are actually gone from the dashboard, and the tab hidden.
    assert v2._chart._dash.is_visible(PANEL_CHART) is False
    assert v2._chart._dash.is_visible(PANEL_NEWS) is True
    assert v2._tabs.isTabVisible(v2._watchlist_tab_index) is False
    v2.shutdown()


def test_grid_cells_round_trip(tmp_path):
    settings = _settings(tmp_path)

    v1 = _build(settings)
    dash = v1._chart._dash
    # Move the news card to the top-left and reflow, then commit (as a drop would).
    dash._items[PANEL_NEWS].x, dash._items[PANEL_NEWS].y = 0, 0
    resolve(list(dash._items.values()), PANEL_NEWS)
    dash.layoutChanged.emit()
    saved = {k: it.as_tuple() for k, it in dash._items.items()}
    v1.shutdown()

    v2 = _build(settings)
    dash2 = v2._chart._dash
    for key, cell in saved.items():
        assert dash2._items[key].as_tuple() == cell   # every card restored to its cell
    v2.shutdown()


def test_hidden_card_restores_to_grid_when_reshown(tmp_path):
    settings = _settings(tmp_path)
    v = _build(settings)
    dash = v._chart._dash
    assert dash.is_visible(PANEL_HEATMAP) is True
    v._chart.set_panel_visible(PANEL_HEATMAP, False)
    assert dash.is_visible(PANEL_HEATMAP) is False
    v._chart.set_panel_visible(PANEL_HEATMAP, True)
    assert dash.is_visible(PANEL_HEATMAP) is True     # comes back onto the grid
    v.shutdown()


def test_grid_dashboard_drag_commits_and_reflows():
    """The drag handlers (not just the engine) move a card to its target cell and
    push the overlapped card out of the way, emitting layoutChanged."""
    from PySide6.QtCore import QPoint
    from PySide6.QtWidgets import QLabel

    from src.ui.widgets.grid_dashboard import GridDashboard

    dash = GridDashboard()
    dash.resize(1200, 600)
    dash.add_panel("a", "A", QLabel("a"), 6, 6, x=0, y=0)
    dash.add_panel("b", "B", QLabel("b"), 6, 6, x=6, y=0)

    fired = []
    dash.layoutChanged.connect(lambda: fired.append(1))

    dash._on_drag_start("a", QPoint(0, 0))
    dash._drag["target"] = (6, 0)          # as if dragged onto b's column
    dash._on_drag_end("a")

    assert dash._items["a"].as_tuple() == (6, 0, 6, 6)   # committed to target
    assert dash._items["b"].y == 6                        # b pushed straight down
    assert fired                                          # persistence signal fired


def test_grid_dashboard_resize_from_left_edge():
    """Dragging the left (west) edge moves the origin and widens the card — proving
    resize works from an edge, not just the bottom-right corner."""
    from PySide6.QtCore import QPoint
    from PySide6.QtWidgets import QLabel

    from src.ui.widgets.grid_dashboard import GridDashboard

    dash = GridDashboard()
    dash.resize(1200, 600)  # 12 cols → 100px per column
    dash.add_panel("a", "A", QLabel("a"), 6, 6, x=3, y=0)   # room on the left

    dash._on_resize_start("a", "w", QPoint(300, 100))
    dash._on_resize_move("a", "w", QPoint(100, 100))        # left edge −200px = −2 cols
    dash._on_resize_end("a", "w")

    it = dash._items["a"]
    assert it.x == 1 and it.w == 8 and it.y == 0            # origin moved left, wider


def test_grid_dashboard_resize_from_bottom_edge():
    """Dragging the bottom (south) edge changes only the height."""
    from PySide6.QtCore import QPoint
    from PySide6.QtWidgets import QLabel

    from src.ui.widgets.grid_dashboard import GridDashboard

    dash = GridDashboard()
    dash.resize(1200, 600)
    dash.add_panel("a", "A", QLabel("a"), 6, 4, x=0, y=0)
    row_h = dash._row_h()

    dash._on_resize_start("a", "s", QPoint(100, 100))
    dash._on_resize_move("a", "s", QPoint(100, int(100 + 2 * row_h)))  # +2 rows
    dash._on_resize_end("a", "s")

    it = dash._items["a"]
    assert it.x == 0 and it.w == 6 and it.h == 6            # only height grew


def test_watchlist_reorder_and_shared_feed():
    """Drag-reorder rewrites the persisted watchlist order, and the shared-quote
    feed populates the cockpit (selects a symbol) without the watchlist's own poll."""
    from src.data.market_data import Quote
    from src.ui.watchlist import WatchlistStore
    from src.ui.watchlist_panel import WatchlistPanel

    class _Fake:
        def __init__(self):
            self.d = {}

        def value(self, k, default=None, type=None):  # noqa: A002
            return self.d.get(k, default)

        def setValue(self, k, v):
            self.d[k] = v

    store = WatchlistStore(settings=_Fake(), seed=False)
    for s in ("AAA", "BBB", "CCC"):
        store.add(s)
    wp = WatchlistPanel(store=store)

    wp._on_reorder(0, 2)                                  # drag AAA to the bottom
    assert store.symbols() == ["BBB", "CCC", "AAA"]

    wp.feed_shared_quotes({"BBB": Quote("BBB", last=10.0, prev_close=9.0, change_pct=0.11)})
    assert wp._chart.selected() == "BBB"                 # first symbol auto-selected
    wp.shutdown()


def test_day_pnl_uses_invested_not_total_capital(tmp_path):
    """Day P&L must be computed on the invested amount (capital - cash), since cash
    has no day move and capital is the total account value."""
    from types import SimpleNamespace

    from src.data.market_data import Quote

    v = _build(_settings(tmp_path))
    v.set_portfolio(SimpleNamespace(
        tickers=["AAPL"], weights={"AAPL": 1.0}, cost_basis={},
        capital=100_000.0, cash=30_000.0,
    ))
    v.set_quotes({"AAPL": Quote("AAPL", last=101.0, prev_close=100.0,
                                change=1.0, change_pct=0.01)})
    assert v._stats["Day P&L"].text() == "$700"       # 70k invested * 1%
    v.shutdown()
