"""App-shell widgets: grouped nav, KPI strip, top bar, data status."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pandas as pd
import pytest

try:
    from PySide6 import QtWidgets

    _APP = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
except Exception as exc:  # pragma: no cover
    pytest.skip(f"Qt unavailable: {exc}", allow_module_level=True)

from datetime import date  # noqa: E402
from types import SimpleNamespace  # noqa: E402

from src.ui import theme  # noqa: E402
from src.ui.widgets.app_shell import (  # noqa: E402
    KpiStrip, NavPanel, TopBar, data_status_text,
)


def test_nav_emits_pages_and_actions():
    nav = NavPanel()
    pages, actions = [], []
    nav.pageSelected.connect(pages.append)
    nav.actionTriggered.connect(actions.append)
    nav._on_clicked(nav._items["Risk"])
    nav._on_clicked(nav._items["@live"])
    assert pages == ["Risk"] and actions == ["live"]
    assert nav.current() == "Risk"  # tool entries don't steal the page highlight


def test_nav_macro_hidden_until_available():
    nav = NavPanel()
    assert nav._items["Macro"].isHidden()
    nav.set_available("Macro", True)
    assert not nav._items["Macro"].isHidden()


def test_nav_covers_every_results_page():
    expected = {"Overview", "Performance", "Risk", "Attribution", "Income", "Optimization",
                "Forecast", "Fundamentals", "News", "Macro", "Data"}
    assert expected <= set(NavPanel().keys())


def test_kpi_strip_tone_and_retheme():
    k = KpiStrip()
    assert len(k._cells) == 8
    k.set_metric("Max drawdown", "−34.5%", "Bench −20%", "neg")
    assert k.cell("Max drawdown").value.text() == "−34.5%"
    assert theme.ACTIVE.red in k.cell("Max drawdown").value.styleSheet()
    k.retheme()
    assert k.cell("Max drawdown").value.text() == "—"


def test_top_bar_context_and_running():
    tb = TopBar()
    tb.set_context("Growth Core", "SPY", date(2022, 1, 1), date(2026, 7, 1), "Rf 4.00%")
    assert tb.chip_period.text() == "Jan 2022 – Jul 2026"
    tb.set_running(True)
    assert not tb.btn_run.isEnabled()


def test_data_status_text():
    idx = pd.bdate_range("2025-01-01", periods=10)
    res = SimpleNamespace(prices=pd.DataFrame({"A": range(10)}, index=idx),
                          risk_free=SimpleNamespace(rate=0.028, source="^IRX period average"),
                          warnings=["x"])
    txt = data_status_text(res)
    assert "as of 14 Jan 2025" in txt and "10 trading days" in txt
    assert "Rf 2.80% (^IRX period average)" in txt and "1 warning" in txt
    assert "no analysis run yet" in data_status_text(None)


def test_graphite_is_default_and_flat():
    assert theme.DEFAULT_THEME == "graphite"
    qss = theme.stylesheet(theme.THEMES["graphite"])
    assert "qlineargradient" not in qss
    assert "#navPanel" in qss and "#kpiStrip" in qss
