"""Overview — growth vs benchmark with drawdown, period returns, a risk snapshot,
the holdings table, outperformance and the strategy summary."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.analytics.summary import holdings_summary, period_returns
from src.charts import plotly_charts as charts

from ..widgets.html_table import Col, rich_table_html
from .web_tab import WebTab


def _pct(v) -> str:
    return f"{v:.2%}" if v is not None and not np.isnan(v) else "—"


def _ratio(v) -> str:
    return f"{v:.2f}" if v is not None and not np.isnan(v) else "—"


HOLDING_COLS = [
    Col("Ticker", "Security", "sym"),
    Col("Weight", "Weight", "pct", 1, bar=True),
    Col("Value", "Value", "money"),
    Col("Price", "Price", "num"),
    Col("RiskContribution", "Risk contrib.", "pct", 1),
    Col("Beta", "Beta", "ratio"),
    Col("Return", "Return", "spct", 1),
    Col("Contribution", "Contrib.", "spct", 1),
]


class OverviewTab(WebTab):
    def _populate(self, results) -> None:
        interp = results.interpretations or {}
        self.add_interpretation(interp.get("performance"))
        bench_name = results.config.benchmark

        # ── Growth (+ underwater drawdown) | period returns + risk snapshot ──
        growth_series: dict[str, pd.Series] = {}
        for key, ps in (
            ("Portfolio", results.active),
            (bench_name or "Benchmark", results.passive),
            ("ORP", results.orp),
            ("HRP", results.hrp),
            ("Rebalanced", results.rebalanced),
            ("Complete", results.complete),
        ):
            if ps is not None:
                growth_series[key] = ps.values
        left = ""
        if growth_series:
            g = charts.growth_chart(growth_series, results.config.capital, focus=True)
            left = self.chart_html(g, 330)
            dd_src = {k: v for k, v in list(growth_series.items())[:2]}
            dd = charts.drawdown_chart(dd_src)
            dd.update_layout(showlegend=False, margin=dict(t=4, b=4), updatemenus=[])
            dd.update_yaxes(title_text="", nticks=3)
            dd.update_xaxes(showticklabels=False)
            left += self.chart_html(dd, 110)
            left = self.panel_html(
                f"Growth of {charts._dollar_format(results.config.capital)}", left,
                explain="growth_chart", aside="Click legend to add strategies")

        right = ""
        if results.active is not None:
            pr = period_returns(results.active.values,
                                results.passive.values if results.passive else None)
            right += self.panel_html("Period returns", rich_table_html(
                pr, [Col("Period", "", "label"), Col("Portfolio", "Port.", "spct"),
                     Col("Benchmark", "Bench", "spct"), Col("Excess", "Excess", "spct")],
                sortable=False, copy=False, table_id="periods"), aside="total, %")
            tr = results.tail_risk or {}
            cap = results.capture_metrics or {}
            snap = pd.DataFrame([
                ("VaR 95% (1d, hist.)", _pct(tr.get("VaR_95", np.nan))),
                ("CVaR 95% (1d)", _pct(tr.get("CVaR_95", np.nan))),
                ("Sortino", _ratio(tr.get("Sortino", np.nan))),
                ("Calmar", _ratio(tr.get("Calmar", np.nan))),
                ("Up / down capture", f"{_pct(cap.get('up_capture', np.nan))} / "
                                      f"{_pct(cap.get('down_capture', np.nan))}"),
                ("Hit rate vs bench", _pct(cap.get("batting_avg", np.nan))),
            ], columns=["Metric", "Value"])
            right += self.panel_html("Risk snapshot", rich_table_html(
                snap, [Col("Metric", "", "label"), Col("Value", "", "label")],
                sortable=False, copy=False, table_id="risksnap"), explain="tail_risk_metrics")
        if left or right:
            self.add_split([(left, 2.6), (right, 1.0)] if right else [(left, 1)])

        # ── Holdings ──
        hs = holdings_summary(results.holdings, results.prices, bench_name)
        if not hs.empty:
            body, total = hs.iloc[:-1], hs.iloc[-1].to_dict()
            self.add_heading("Holdings")
            self.add_html(f"<div class='blurb'>{len(body)} positions · current weights · "
                          "risk contribution and beta over the last 252 trading days · "
                          "return and contribution since purchase</div>")
            self.add_rich_table(body, HOLDING_COLS, total=total)

        # ── Outperformance ──
        if results.active and results.passive:
            self.add_chart(
                charts.outperformance_chart(results.active.values, results.passive.values),
                height=300,
                explain="outperformance_chart",
            )

        # ── Strategy summary ──
        if results.active and results.passive:
            self.add_heading("Strategy summary", explain="performance_table")
            rows = []
            for ps in (results.active, results.passive, results.orp, results.hrp,
                       results.rebalanced, results.complete):
                if ps is None:
                    continue
                rows.append({
                    "Portfolio": ps.name,
                    "Ann. Return": ps.ann_return,
                    "Ann. Volatility": ps.ann_vol,
                    "Sharpe": ps.sharpe,
                    "Max Drawdown": ps.max_dd,
                })
            self.add_rich_table(pd.DataFrame(rows), [
                Col("Portfolio", "Portfolio", "label"),
                Col("Ann. Return", "Ann. return", "spct"),
                Col("Ann. Volatility", "Ann. vol", "pct"),
                Col("Sharpe", "Sharpe", "ratio"),
                Col("Max Drawdown", "Max drawdown", "spct"),
            ])
