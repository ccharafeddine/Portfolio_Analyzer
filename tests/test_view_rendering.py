"""Render the analysis pages from a real (offline) pipeline run and check the
Phase 3/4 sections are present, well-formed and use the Graphite tables."""

import re

import pytest

pytest.importorskip("PySide6.QtWebEngineWidgets")
from PySide6.QtWidgets import QApplication  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _html(tab_cls, results, app):
    tab = tab_cls()
    tab.set_results(results)
    tab._populate(results)
    return tab._build_page()


def test_pipeline_produced_phase3_results(offline_results):
    r = offline_results
    assert r.var_table is not None and len(r.var_table) == 4
    assert r.drawdown_table is not None and not r.drawdown_table.empty
    assert r.sector_attribution is not None and not r.sector_attribution.empty
    assert r.beta_shocks is not None and r.factor_models
    assert r.data_quality is not None and r.timings


def test_risk_page(offline_results, app):
    from src.ui.tabs.risk_tab import RiskTab

    html = _html(RiskTab, offline_results, app)
    for title in ("Value at Risk by method", "VaR backtest", "Largest drawdowns",
                  "Tail risk by portfolio", "Return shape", "Market shocks via beta"):
        assert title in html, title
    assert "MaxDrawdown" not in html and "VAR_95" not in html  # raw column names gone
    assert html.count("<table") >= 5


def test_attribution_page(offline_results, app):
    from src.ui.tabs.attribution_tab import AttributionTab

    html = _html(AttributionTab, offline_results, app)
    assert "Sector attribution" in html and "Allocation vs selection" in html
    assert re.search(r"\d\.\d\d \(-?\d+\.\d\)", html)  # coefficient (t-stat)


def test_performance_page(offline_results, app):
    from src.ui.tabs.performance_tab import PerformanceTab

    assert "Rolling 12-month Sharpe" in _html(PerformanceTab, offline_results, app)


def test_optimization_page(offline_results, app):
    from src.ui.tabs.optimization_tab import OptimizationTab

    html = _html(OptimizationTab, offline_results, app)
    assert "Turnover" in html and "rebalance trades" in html and "id='trades'" in html


def test_offline_run_has_no_step_failures(offline_results):
    assert not [w for w in offline_results.warnings if "failed" in w.lower()], offline_results.warnings


def test_data_page_builds(offline_results, app):
    from src.ui.tabs.data_tab import DataTab

    tab = DataTab()
    tab.set_results(offline_results)
    tab._populate(offline_results)
    from PySide6.QtWidgets import QLabel, QPushButton

    labels = " | ".join(lbl.text() for lbl in tab.findChildren(QLabel))
    assert "Data quality" in labels and "Run time" in labels
    names = {b.text(): b.objectName() for b in tab.findChildren(QPushButton)}
    assert names["Export HTML report"] != "secondary" and names["PDF"] == "secondary"
