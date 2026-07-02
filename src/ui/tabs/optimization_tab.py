"""Optimization tab — efficient frontier, ORP/HRP weights, dendrogram,
concentration metrics, weight drift, rebalancing comparison, risk contribution.

Transcribed from app.py's ``with tab_optimization:`` block (lines 790-926).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.charts import plotly_charts as charts

from .web_tab import WebTab


def _pct(v) -> str:
    return f"{v:.2%}" if v is not None and not np.isnan(v) else "—"


def _ratio(v, d: int = 2) -> str:
    return f"{v:.{d}f}" if v is not None and not np.isnan(v) else "—"


class OptimizationTab(WebTab):
    def _populate(self, results) -> None:
        interp = results.interpretations or {}
        self.add_interpretation(interp.get("optimization"))

        orp_opt = results.orp_optimization
        if orp_opt is None:
            self.add_interpretation("ORP optimization was not included in this run.")
            return

        cfg = results.config
        rets_m = results.monthly_returns
        asset_cols = [
            t for t in cfg.tickers if t in rets_m.columns and t != cfg.benchmark
        ]
        asset_mu = ((1 + rets_m[asset_cols].mean()) ** 12 - 1) if asset_cols else None
        asset_vol = (rets_m[asset_cols].std() * np.sqrt(12)) if asset_cols else None

        # Efficient frontier
        self.add_chart(
            charts.efficient_frontier_chart(
                orp_opt.frontier_vols,
                orp_opt.frontier_returns,
                orp_opt.expected_vol,
                orp_opt.expected_return,
                cfg.risk_free_rate,
                asset_vols=asset_vol,
                asset_returns=asset_mu,
            ),
            height=440,
            explain="efficient_frontier",
        )

        # ORP weights + complete-portfolio donut
        weights_row = [charts.weights_bar(orp_opt.weights, "ORP Weights (Max-Sharpe)")]
        weights_keys = ["orp_weights"]
        if cfg.include_complete:
            cp_weights = {
                t: float(cfg.complete_portfolio.y) * float(w)
                for t, w in orp_opt.weights.items()
                if abs(w) > 1e-6
            }
            cp_weights["Risk-Free"] = 1.0 - cfg.complete_portfolio.y
            weights_row.append(
                charts.allocation_donut(
                    cp_weights, f"Complete Portfolio (y={cfg.complete_portfolio.y:.0%})"
                )
            )
            weights_keys.append("complete_portfolio")
        self.add_chart_row(weights_row, height=360, explains=weights_keys)

        # HRP weights + dendrogram
        if results.hrp_weights is not None:
            hrp_row = [charts.weights_bar(results.hrp_weights, "HRP Weights")]
            hrp_keys = ["hrp_weights"]
            if results.hrp_linkage is not None:
                hrp_row.append(charts.dendrogram_chart(results.hrp_linkage, asset_cols))
                hrp_keys.append("dendrogram")
            self.add_chart_row(hrp_row, height=360, explains=hrp_keys)

        # Concentration metrics
        from src.analytics.risk import (
            concentration_ratio,
            effective_n_bets,
            herfindahl_index,
        )

        self.add_heading("Concentration Metrics", explain="concentration")
        if results.holdings is not None and "RealizedWeight" in results.holdings:
            aw = results.holdings["RealizedWeight"].values
            self.add_stat_grid(
                [
                    ("Active · HHI", f"{herfindahl_index(aw):.4f}"),
                    ("Active · Eff. N Bets", f"{effective_n_bets(aw):.1f}"),
                    ("Active · Top-3 Conc.", f"{concentration_ratio(aw, 3):.2%}"),
                ],
                columns=3,
            )
        ow = orp_opt.weights.values
        self.add_stat_grid(
            [
                ("ORP · HHI", f"{herfindahl_index(ow):.4f}"),
                ("ORP · Eff. N Bets", f"{effective_n_bets(ow):.1f}"),
                ("ORP · Top-3 Conc.", f"{concentration_ratio(ow, 3):.2%}"),
            ],
            columns=3,
        )

        # Weight drift
        if results.weight_drift is not None and not results.weight_drift.empty:
            self.add_chart(
                charts.weight_drift_chart(results.weight_drift), height=360, explain="weight_drift"
            )

        # Rebalanced vs buy-and-hold
        if results.rebalanced is not None and results.active is not None:
            self.add_heading("Rebalanced vs Buy-and-Hold", explain="rebalancing")
            rows = [
                {
                    "Strategy": ps.name,
                    "Ann. Return": _pct(ps.ann_return),
                    "Ann. Volatility": _pct(ps.ann_vol),
                    "Sharpe": _ratio(ps.sharpe),
                    "Max Drawdown": _pct(ps.max_dd),
                }
                for ps in (results.active, results.rebalanced)
            ]
            self.add_table(pd.DataFrame(rows))

        # Turnover
        if results.turnover_table is not None and not results.turnover_table.empty:
            self.add_heading("Quarterly Turnover", explain="turnover")
            self.add_table(results.turnover_table)

        # ORP stats
        self.add_heading("Optimal Risky Portfolio Statistics", explain="orp_stats")
        self.add_table(
            pd.DataFrame(
                [
                    {
                        "Expected Return": _pct(orp_opt.expected_return),
                        "Expected Volatility": _pct(orp_opt.expected_vol),
                        "Sharpe Ratio": _ratio(orp_opt.sharpe, 3),
                    }
                ]
            )
        )

        # Risk contribution
        if results.risk_contribution is not None:
            self.add_chart(
                charts.risk_contribution_chart(
                    results.risk_contribution, "ORP Risk Contribution by Asset"
                ),
                height=360,
                explain="risk_contribution",
            )
