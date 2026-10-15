"""Risk tab — drawdown, VaR distribution, correlation, rolling metrics,
tail-risk statistics, and stress testing.

Transcribed from app.py's ``with tab_risk:`` block (lines 532-628).
"""

from __future__ import annotations

import numpy as np

import pandas as pd

from src.charts import plotly_charts as charts
from src.data import transforms as T

from .. import theme
from ..widgets.html_table import Col, fmt_value, rich_table_html
from .web_tab import WebTab

_VAR_COLS = [Col("Method", "Method", "label"), Col("VaR 95%", "VaR 95%", "spct"),
             Col("ES 95%", "ES 95%", "spct"), Col("VaR 99%", "VaR 99%", "spct"),
             Col("ES 99%", "ES 99%", "spct")]
_BT_COLS = [Col("Method", "Method", "label"), Col("Days", "Days", "int"),
            Col("Breaches", "Breaches", "int"), Col("Expected", "Expected", "num", 1),
            Col("Breach rate", "Rate", "pct"), Col("Kupiec p", "Kupiec p", "num", 3),
            Col("Verdict", "Verdict", "text")]
_DD_COLS = [Col("Peak", "Peak", "text"), Col("Trough", "Trough", "text"),
            Col("Recovery", "Recovery", "text"), Col("Depth", "Depth", "spct", 1),
            Col("Decline days", "Decline", "int"), Col("Recovery days", "Recover", "int"),
            Col("Length", "Length (d)", "int")]
_SHOCK_COLS = [Col("Market shock", "Market move", "spct", 0), Col("Portfolio", "Portfolio", "spct", 1),
               Col("P&L", "P&L", "money")]


def _date(v) -> str:
    return "Open" if v is None or pd.isna(v) else pd.Timestamp(v).strftime("%b %d, %Y")


class RiskTab(WebTab):
    def _populate(self, results) -> None:
        interp = results.interpretations or {}
        self.add_interpretation(interp.get("risk"))

        # Drawdown chart
        dd_series = {}
        for key, ps in (("Active", results.active), ("Passive", results.passive), ("ORP", results.orp)):
            if ps is not None:
                dd_series[key] = ps.values
        if dd_series:
            self.add_chart(charts.drawdown_chart(dd_series), height=380, explain="drawdown_chart")

        # VaR histogram + correlation heatmap, side by side
        row_figs, row_keys = [], []
        if results.active:
            var95, cvar95 = T.var_cvar(results.active.daily_returns, 0.95)
            row_figs.append(
                charts.return_distribution_chart(
                    results.active.daily_returns,
                    var_95=var95,
                    cvar_95=cvar95,
                    title="Active Portfolio: Daily Returns",
                )
            )
            row_keys.append("return_distribution")
        if results.correlation_matrix is not None:
            row_figs.append(charts.correlation_heatmap(results.correlation_matrix))
            row_keys.append("correlation_heatmap")
        self.add_chart_row(row_figs, height=360, explains=row_keys)

        # Correlation regime
        if results.correlation_regime is not None and not results.correlation_regime.empty:
            self.add_chart(
                charts.correlation_regime_chart(results.correlation_regime),
                height=340,
                explain="correlation_regime",
            )

        # Rolling metrics
        if not results.monthly_returns.empty:
            asset_cols = [
                t for t in results.config.tickers if t in results.monthly_returns.columns
            ]
            if asset_cols and len(results.monthly_returns) >= 12:
                self.add_chart(
                    charts.rolling_metrics_chart(
                        results.monthly_returns[asset_cols],
                        window=12,
                        rf_annual=results.config.risk_free_rate,
                    ),
                    height=380,
                    explain="rolling_metrics",
                )

        # VaR / ES by method + backtest of breaches
        vt = getattr(results, "var_table", None)
        bts = getattr(results, "var_backtest", None)
        if vt is not None and not vt.empty:
            left = self.panel_html("Value at Risk by method (1-day)",
                                   rich_table_html(vt, _VAR_COLS, sortable=False, table_id="var"),
                                   explain="var_methods", aside="Active portfolio")
            right = ""
            if bts is not None and not bts.empty:
                right = self.panel_html("VaR backtest, 99%, 250-day window",
                                        rich_table_html(bts, _BT_COLS, sortable=False, table_id="vbt"),
                                        explain="var_backtest")
            self.add_split([(left, 1), (right, 1.15)] if right else [(left, 1)])
        ser = getattr(results, "var_backtest_series", None)
        if ser is not None and not ser.empty:
            n = int(ser["breach"].sum())
            self.add_html(self.panel_html(
                "Daily returns vs historical VaR 99%",
                self.chart_html(charts.var_backtest_chart(ser, breach_color=theme.ACTIVE.red), 300),
                aside=f"{n} breaches in {len(ser)} days"))

        # Top drawdowns
        ddt = getattr(results, "drawdown_table", None)
        if ddt is not None and not ddt.empty:
            disp = ddt.copy()
            for c in ("Peak", "Trough", "Recovery"):
                disp[c] = disp[c].map(_date)
            self.add_html(self.panel_html("Largest drawdowns",
                                          rich_table_html(disp, _DD_COLS, table_id="dd"),
                                          explain="drawdown_table", aside="Trading days"))

        # Tail metrics by portfolio | extended statistics, side by side
        dm = results.drawdown_metrics
        tr = results.tail_risk or {}
        if dm is not None and not dm.empty:
            left = self.panel_html("Tail risk by portfolio", rich_table_html(
                dm, [Col("Portfolio", "Portfolio", "label"), Col("MaxDrawdown", "Max DD", "spct", 1),
                     Col("VaR_95", "VaR 95%", "spct"), Col("CVaR_95", "ES 95%", "spct"),
                     Col("VaR_99", "VaR 99%", "spct"), Col("CVaR_99", "ES 99%", "spct")],
                sortable=False, table_id="tail"), explain="tail_risk_metrics",
                aside="historical, 1-day")
            parts = [(left, 1.4)]
            if tr:
                stats = pd.DataFrame([
                    ("Sortino", tr.get("Sortino", np.nan), "ratio"),
                    ("Calmar", tr.get("Calmar", np.nan), "ratio"),
                    ("Gain-to-pain", tr.get("Gain_to_Pain", np.nan), "ratio"),
                    ("Skewness", tr.get("Skewness", np.nan), "ratio"),
                    ("Excess kurtosis", tr.get("Excess_Kurtosis", np.nan), "ratio"),
                    ("Worst day", tr.get("Worst_Day", np.nan), "spct"),
                    ("Best day", tr.get("Best_Day", np.nan), "spct"),
                ], columns=["Stat", "Value", "kind"])
                stats["Text"] = [fmt_value(v, k) for v, k in zip(stats["Value"], stats["kind"])]
                parts.append((self.panel_html("Return shape", rich_table_html(
                    stats, [Col("Stat", "Statistic", "label"), Col("Text", "Active", "text")],
                    sortable=False, copy=False, table_id="shape"),
                    explain="extended_risk"), 1))
            self.add_split(parts)

        # Stress testing (the chart's own title + '?' serves as the heading)
        if results.stress_df is not None and not results.stress_df.empty:
            self.add_chart(
                charts.stress_test_chart(results.stress_df), height=360, explain="stress_test"
            )
            srows = [{"Scenario": r.name, "Period": f"{r.start} → {r.end}",
                      "Portfolio": r.portfolio_return, "Benchmark": r.benchmark_return,
                      "Excess": r.portfolio_return - r.benchmark_return,
                      "MaxDD": r.max_drawdown, "Days": r.days if r.data_available else np.nan,
                      "Basis": results.stress_df["Basis"].iloc[i]
                      if "Basis" in results.stress_df else ""}
                     for i, r in enumerate(results.stress_results)]
            self.add_rich_table(pd.DataFrame(srows), [
                Col("Scenario", "Scenario", "label"), Col("Period", "Period", "text"),
                Col("Portfolio", "Portfolio", "spct", 1), Col("Benchmark", "Benchmark", "spct", 1),
                Col("Excess", "Excess", "spct", 1), Col("MaxDD", "Max DD", "spct", 1),
                Col("Days", "Days", "int"), Col("Basis", "Basis", "text")])
            if (results.stress_df.get("Basis", pd.Series(dtype=str)).astype(str)
                    .str.startswith("Replay").any()):
                self.add_interpretation(
                    "Replay rows apply today's weights to each holding's own price history, "
                    "because the scenario predates the analysis window. They show how the "
                    "current mix would have behaved, not what the account actually did.")
        bs = getattr(results, "beta_shocks", None)
        if bs is not None and not bs.empty:
            beta = float(bs["Beta"].iloc[0])
            self.add_html(self.panel_html(
                "Market shocks via beta", rich_table_html(bs, _SHOCK_COLS, sortable=False,
                                                         table_id="shock"),
                explain="beta_shocks", aside=f"Daily beta {beta:.2f} vs benchmark"))

        # Interactive what-if scenario builder (client-side; recomputes live)
        model = getattr(results, "scenario_model", None)
        if model and model.get("drivers"):
            from .scenario_section import scenario_html

            self.add_heading("Scenario Analysis", explain="scenario")
            self.add_interpretation(
                "Estimate how your portfolio would move under hypothetical shocks. Edit "
                "the inputs (or pick a preset) and the impact updates live. Macro factors "
                "use your portfolio's estimated sensitivity; holdings use their weight."
            )
            self.add_html(scenario_html(model))
