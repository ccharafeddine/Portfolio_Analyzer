"""Performance tab — the benchmark-relative measurement suite.

Shows the data-coverage timeline (so the inception handling is transparent), a
dual-window performance summary, up/down capture + batting average, rolling
alpha/beta, and the backtest's turnover/cost summary.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.analytics.summary import monthly_calendar
from src.charts import plotly_charts as charts

from .. import theme
from ..widgets.html_table import heatmap_table_html
from .web_tab import WebTab

# Metric name -> how to format its value.
_PCT_METRICS = {
    "Ann. Return", "Ann. Volatility", "Max Drawdown", "Tracking Error",
    "Batting Avg", "Alpha (ann.)",
}
_CAP_METRICS = {"Up Capture", "Down Capture"}  # ratios shown as %


def _fmt(metric: str, v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    if metric in _PCT_METRICS:
        return f"{v:.2%}"
    if metric in _CAP_METRICS:
        return f"{v * 100:.0f}%"
    return f"{v:.2f}"


class PerformanceTab(WebTab):
    def _populate(self, results) -> None:
        interp = results.interpretations or {}
        self.add_interpretation(interp.get("performance"))

        bt = results.backtest

        # Monthly return calendar (portfolio; benchmark calendar-year in grey).
        if results.active is not None:
            cal = monthly_calendar(results.active.values,
                                   results.passive.values if results.passive else None)
            if not cal.empty:
                t = theme.ACTIVE
                self.add_heading("Monthly returns")
                self.add_html(heatmap_table_html(cal, t.green, t.red))

        # Coverage timeline — makes the inception handling visible.
        if bt is not None and bt.coverage:
            end = results.active.values.index[-1] if results.active is not None else None
            self.add_chart(charts.coverage_timeline_chart(bt.coverage, end), height=260, explain="coverage")

        # Dual-window summary table.
        ps = results.performance_summary
        if ps is not None and not ps.empty:
            value_cols = [c for c in ps.columns if c != "Metric"]
            heading_key = "dual_window" if "Common Window" in value_cols else "performance_table"
            self.add_heading("Benchmark-relative statistics", explain=heading_key)
            disp = ps.copy()
            for col in value_cols:
                disp[col] = [_fmt(m, v) for m, v in zip(disp["Metric"], disp[col])]
            self.add_table(disp)

        # Capture + batting stat grid.
        cap = results.capture_metrics
        if cap:
            self.add_heading("Benchmark capture", explain="capture")
            self.add_stat_grid(
                [
                    ("Up Capture", _fmt("Up Capture", cap.get("up_capture"))),
                    ("Down Capture", _fmt("Down Capture", cap.get("down_capture"))),
                    ("Capture Ratio", _fmt("Capture Ratio", cap.get("capture_ratio"))),
                    ("Batting Avg", _fmt("Batting Avg", cap.get("batting_avg"))),
                ],
                columns=4,
            )

        # Rolling 12-month Sharpe, portfolio vs benchmark.
        if results.active is not None and results.passive is not None:
            from src.analytics import metrics as M

            rf = results.config.risk_free_rate
            series = {}
            for name, ps in (("Portfolio", results.active), ("Benchmark", results.passive)):
                m = ps.values.resample("ME").last().pct_change().dropna()
                series[name] = M.rolling_sharpe(m, window=12, rf_annual=rf, periods_per_year=12)
            if any(len(s.dropna()) for s in series.values()):
                self.add_html(self.panel_html(
                    "Rolling 12-month Sharpe",
                    self.chart_html(charts.rolling_sharpe_chart(series), 300),
                    explain="rolling_sharpe", aside="Monthly returns, excess of risk-free"))

        # Rolling alpha / beta.
        rab = results.rolling_alpha_beta
        if rab is not None and not rab.empty:
            self.add_chart(charts.rolling_alpha_beta_chart(rab), height=440, explain="rolling_alpha_beta")

        # Tax analysis.
        tm = results.tax_metrics
        if tm:
            self.add_heading("Tax analysis", explain="tax")
            self.add_stat_grid(
                [
                    ("Unrealized Gain", f"${tm.get('unrealized_gain', 0):,.0f}"),
                    ("Harvestable Losses", f"${tm.get('harvest_potential', 0):,.0f}"),
                    ("Loss Candidates", f"{tm.get('n_harvest', 0)}"),
                    ("Est. Tax (realized)", f"${tm.get('estimated_tax', 0):,.0f}"),
                ],
                columns=4,
            )
            td = results.tax_detail
            if td is not None and not td.empty:
                disp = pd.DataFrame(
                    {
                        "Ticker": td["Ticker"],
                        "Shares": td["Shares"].map(lambda v: f"{v:,.1f}"),
                        "Avg Cost": td["AvgCost"].map(lambda v: f"${v:,.2f}" if v == v else "—"),
                        "Price": td["CurrentPrice"].map(lambda v: f"${v:,.2f}" if v == v else "—"),
                        "Market Value": td["MarketValue"].map(lambda v: f"${v:,.0f}"),
                        "Unrealized": td["UnrealizedGain"].map(lambda v: f"${v:,.0f}" if v == v else "—"),
                        "Return": td["Return"].map(lambda v: f"{v:.1%}" if v == v else "—"),
                    }
                )
                self.add_table(disp)

        # Turnover & cost summary from the engine.
        if bt is not None:
            n_rebal = len(bt.rebalance_dates)
            avg_turn = (
                bt.turnover_table["Turnover"].mean()
                if bt.turnover_table is not None and not bt.turnover_table.empty
                else np.nan
            )
            self.add_heading("Trading and costs", explain="trading_costs")
            self.add_stat_grid(
                [
                    ("Rebalances", f"{n_rebal}"),
                    ("Avg Turnover", _fmt("Ann. Return", avg_turn) if not np.isnan(avg_turn) else "—"),
                    ("Total Costs", f"${bt.total_costs:,.0f}"),
                    ("Inception Mode", bt.inception_mode.title()),
                ],
                columns=4,
            )
