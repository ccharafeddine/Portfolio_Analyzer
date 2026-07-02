"""Overview tab — growth, outperformance, and a performance-summary table.

Transcribed from app.py's ``with tab_overview:`` block (lines 478-525).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.charts import plotly_charts as charts

from .web_tab import WebTab


def _pct(v) -> str:
    return f"{v:.2%}" if v is not None and not np.isnan(v) else "—"


def _ratio(v) -> str:
    return f"{v:.2f}" if v is not None and not np.isnan(v) else "—"


class OverviewTab(WebTab):
    def _populate(self, results) -> None:
        interp = results.interpretations or {}
        self.add_interpretation(interp.get("performance"))

        # Growth chart — active/passive/orp/hrp/rebalanced/complete
        growth_series: dict[str, pd.Series] = {}
        for key, ps in (
            ("Active", results.active),
            ("Passive", results.passive),
            ("ORP", results.orp),
            ("HRP", results.hrp),
            ("Rebalanced", results.rebalanced),
            ("Complete", results.complete),
        ):
            if ps is not None:
                growth_series[key] = ps.values
        if growth_series:
            self.add_chart(
                charts.growth_chart(growth_series, results.config.capital),
                height=440,
                explain="growth_chart",
            )

        # Outperformance
        if results.active and results.passive:
            self.add_chart(
                charts.outperformance_chart(results.active.values, results.passive.values),
                height=360,
                explain="outperformance_chart",
            )

        # Performance summary table
        if results.active and results.passive:
            self.add_heading("Performance Summary", explain="performance_table")
            rows = []
            for ps in (
                results.active,
                results.passive,
                results.orp,
                results.hrp,
                results.rebalanced,
                results.complete,
            ):
                if ps is None:
                    continue
                rows.append(
                    {
                        "Portfolio": ps.name,
                        "Ann. Return": _pct(ps.ann_return),
                        "Ann. Volatility": _pct(ps.ann_vol),
                        "Sharpe": _ratio(ps.sharpe),
                        "Max Drawdown": _pct(ps.max_dd),
                    }
                )
            self.add_table(pd.DataFrame(rows))
