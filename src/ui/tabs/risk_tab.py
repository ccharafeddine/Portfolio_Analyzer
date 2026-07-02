"""Risk tab — drawdown, VaR distribution, correlation, rolling metrics,
tail-risk statistics, and stress testing.

Transcribed from app.py's ``with tab_risk:`` block (lines 532-628).
"""

from __future__ import annotations

import numpy as np

from src.charts import plotly_charts as charts
from src.data import transforms as T

from .web_tab import WebTab


def _pct(v) -> str:
    return f"{v:.2%}" if v is not None and not np.isnan(v) else "N/A"


def _num(v, decimals: int = 2) -> str:
    return f"{v:.{decimals}f}" if v is not None and not np.isnan(v) else "N/A"


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

        # Drawdown metrics table
        if results.drawdown_metrics is not None:
            self.add_heading("Tail Risk Metrics", explain="tail_risk_metrics")
            self.add_table(results.drawdown_metrics)

        # Extended risk statistics
        if results.tail_risk:
            tr = results.tail_risk
            self.add_heading("Extended Risk Statistics", explain="extended_risk")
            self.add_stat_grid(
                [
                    ("Sortino Ratio", _num(tr.get("Sortino", np.nan))),
                    ("Calmar Ratio", _num(tr.get("Calmar", np.nan))),
                    ("Skewness", _num(tr.get("Skewness", np.nan), 3)),
                    ("Excess Kurtosis", _num(tr.get("Excess_Kurtosis", np.nan), 3)),
                    ("Worst Day", _pct(tr.get("Worst_Day", np.nan))),
                    ("Best Day", _pct(tr.get("Best_Day", np.nan))),
                    ("Gain-to-Pain", _num(tr.get("Gain_to_Pain", np.nan))),
                    ("Max Drawdown", _pct(tr.get("Max_Drawdown", np.nan))),
                ],
                columns=4,
            )

        # Stress testing (the chart's own title + '?' serves as the heading)
        if results.stress_df is not None and not results.stress_df.empty:
            self.add_chart(
                charts.stress_test_chart(results.stress_df), height=360, explain="stress_test"
            )
            self.add_table(results.stress_df)

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
