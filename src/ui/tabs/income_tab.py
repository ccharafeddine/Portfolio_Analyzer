"""Income tab — dividend income metrics, income by position, cumulative income.

Transcribed from app.py's ``with tab_income:`` block (lines 751-783).
"""

from __future__ import annotations

from src.charts import plotly_charts as charts

from .web_tab import WebTab

_DISPLAY_COLS = [
    "Ticker",
    "Shares",
    "AnnualDividendPerShare",
    "AnnualIncome",
    "YieldOnCost",
    "CurrentYield",
    "IncomeGrowthRate",
]


class IncomeTab(WebTab):
    def _populate(self, results) -> None:
        interp = results.interpretations or {}
        self.add_interpretation(interp.get("income"))

        if results.income_metrics is not None:
            im = results.income_metrics
            self.add_heading("Income Summary", explain="income_metrics")
            self.add_stat_grid(
                [
                    ("Annual Income", f"${im.get('total_annual_income', 0):,.2f}"),
                    ("Portfolio Yield", f"{im.get('portfolio_yield', 0):.2%}"),
                    ("Avg Yield on Cost", f"{im.get('avg_yield_on_cost', 0):.2%}"),
                    ("Dividend Payers", f"{im.get('n_payers', 0)}"),
                ],
                columns=4,
            )

        if results.income_summary is not None and not results.income_summary.empty:
            row = [charts.income_bar_chart(results.income_summary)]
            row_keys = ["income_by_position"]
            if results.cumulative_income is not None and not results.cumulative_income.empty:
                row.append(charts.cumulative_income_chart(results.cumulative_income))
                row_keys.append("cumulative_income")
            self.add_chart_row(row, height=360, explains=row_keys)

            self.add_heading("Dividend Income by Position", explain="income_by_position")
            cols = [c for c in _DISPLAY_COLS if c in results.income_summary.columns]
            self.add_table(results.income_summary[cols])
        else:
            self.add_interpretation(
                "Dividend income data not available. Enable dividend-adjusted income "
                "in the configuration to compute income analytics."
            )
