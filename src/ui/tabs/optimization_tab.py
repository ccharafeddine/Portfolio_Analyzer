"""Optimization tab — efficient frontier, ORP/HRP weights, dendrogram,
concentration metrics, weight drift, rebalancing comparison, risk contribution.

Transcribed from app.py's ``with tab_optimization:`` block (lines 790-926).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.charts import plotly_charts as charts

from ..widgets.html_table import Col, rich_table_html
from .web_tab import WebTab


def _pct(v) -> str:
    return f"{v:.2%}" if v is not None and not np.isnan(v) else "—"


def _ratio(v, d: int = 2) -> str:
    return f"{v:.{d}f}" if v is not None and not np.isnan(v) else "—"


def _usd(v) -> str:
    if v is None or np.isnan(v):
        return "—"
    return f"{'+' if v >= 0 else '−'}${abs(v):,.0f}"


def _shares(v) -> str:
    if v is None or np.isnan(v):
        return "—"
    return f"{'+' if v >= 0 else '−'}{abs(v):,.2f}"


class OptimizationTab(WebTab):
    def _populate(self, results) -> None:
        interp = results.interpretations or {}
        self.add_interpretation(interp.get("optimization"))

        orp_opt = results.orp_optimization
        if orp_opt is None:
            self.add_interpretation("ORP optimization was not included in this run.")
            return

        cfg = results.config

        # Note when Black-Litterman views shaped these weights.
        bl = cfg.black_litterman
        if bl.enabled and bl.views:
            phrases = []
            for v in bl.views:
                if v.type == "relative":
                    phrases.append(
                        f"{v.asset_long} to beat {v.asset_short} by {v.q:.1%} ({v.confidence})"
                    )
                else:
                    phrases.append(f"{v.asset} ≈ {v.q:.1%}/yr ({v.confidence})")
            self.add_interpretation(
                "Black-Litterman is on: the optimizer's expected returns were blended "
                "with your views — " + "; ".join(phrases) + ". The ORP below reflects them."
            )
        rets_m = results.monthly_returns
        asset_cols = [
            t for t in cfg.tickers if t in rets_m.columns and t != cfg.benchmark
        ]
        asset_mu = ((1 + rets_m[asset_cols].mean()) ** 12 - 1) if asset_cols else None
        asset_vol = (rets_m[asset_cols].std() * np.sqrt(12)) if asset_cols else None

        # Efficient frontier | weights by strategy + strategy comparison
        frontier = charts.efficient_frontier_chart(
            orp_opt.frontier_vols,
            orp_opt.frontier_returns,
            orp_opt.expected_vol,
            orp_opt.expected_return,
            cfg.risk_free_rate,
            asset_vols=asset_vol,
            asset_returns=asset_mu,
        )
        left = self.panel_html("Efficient frontier & capital allocation line",
                               self.chart_html(frontier, 430), explain="efficient_frontier")
        right = ""
        wt = self._weights_table(results)
        if not wt.empty:
            cols = [Col("Asset", "Asset", "sym"), Col("Current", "Current", "pct", 1),
                    Col("ORP", "ORP", "pct", 1), Col("Δ ORP", "Δ", "spct", 1)]
            if "HRP" in wt.columns:
                cols.append(Col("HRP", "HRP", "pct", 1))
            right += self.panel_html("Weights by strategy", rich_table_html(
                wt, cols, sortable=True, copy=True, table_id="wts"), aside="target weights")
        sc = self._strategy_table(results)
        if not sc.empty:
            right += self.panel_html("Strategy comparison", rich_table_html(
                sc, [Col("Strategy", "", "label"), Col("Ann. return", "Return", "spct", 1),
                     Col("Ann. vol", "Vol", "pct", 1), Col("Sharpe", "Sharpe", "ratio"),
                     Col("Max DD", "Max DD", "spct", 1), Col("Turnover", "Turnover", "pct", 1),
                     Col("Cost", "Cost", "money")],
                sortable=False, copy=True, table_id="strat"),
                aside=("realized, annualized · turnover/cost to adopt from today at "
                       f"{self._cost_bps(results):g} bps · <a href='#' class='jump' "
                       "onclick=\"document.getElementById('trades').scrollIntoView();"
                       "return false;\">rebalance trades</a>"))
        self.add_split([(left, 1.35), (right, 1.2)] if right else [(left, 1)])

        # The user's actual account allocation — stock weights plus a Cash slice
        # when a cash balance is held (cash as a first-class holding).
        from src.ui.allocation import active_allocation_weights

        active_alloc = active_allocation_weights(
            dict(cfg.weights), cfg.capital, getattr(cfg, "cash", 0.0)
        )
        self.add_heading("Current allocation")
        self.add_chart(
            charts.allocation_donut(active_alloc, ""),
            height=320,
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

        # Trade recommendations (buy/sell to rebalance)
        self.add_html("<div id='trades'></div>")
        self._render_trades(results)

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

    @staticmethod
    def _weights_table(results) -> pd.DataFrame:
        cfg = results.config
        cur = pd.Series(dict(cfg.weights), dtype=float)
        if cur.sum() > 0:
            cur = cur / cur.sum()
        orp = results.orp_optimization.weights if results.orp_optimization else pd.Series(dtype=float)
        df = pd.DataFrame({"Current": cur, "ORP": orp}).fillna(0.0)
        if results.hrp_weights is not None and len(results.hrp_weights):
            df["HRP"] = pd.Series(results.hrp_weights, dtype=float).reindex(df.index).fillna(0.0)
        df["Δ ORP"] = df["ORP"] - df["Current"]
        df = df.sort_values("Current", ascending=False)
        df.index.name = "Asset"
        return df.reset_index()

    @staticmethod
    def _cost_bps(results) -> float:
        """Configured per-side cost; 5 bps illustrative default when unset."""
        bps = float(getattr(results.config.backtest, "transaction_cost_bps", 0.0) or 0.0)
        return bps if bps > 0 else 5.0

    @classmethod
    def _strategy_table(cls, results) -> pd.DataFrame:
        from src.analytics import metrics as M

        cfg = results.config
        reco = results.trade_recos
        current = (pd.Series(reco["CurrentWeight"].values, index=reco["Ticker"])
                   if reco is not None and not reco.empty else None)
        value = float(results.active.values.iloc[-1]) if results.active is not None else 0.0
        targets = {
            "active": pd.Series(dict(cfg.weights), dtype=float),
            "orp": results.orp_optimization.weights if results.orp_optimization else None,
            "hrp": results.hrp_weights,
        }
        bps = cls._cost_bps(results)
        rows = []
        for key, ps in (("active", results.active), ("passive", results.passive),
                        ("orp", results.orp), ("hrp", results.hrp),
                        ("rebalanced", results.rebalanced), ("complete", results.complete)):
            if ps is None:
                continue
            turn = cost = np.nan
            tgt = targets.get(key)
            if current is not None and tgt is not None and len(tgt):
                tgt = pd.Series(tgt, dtype=float)
                tgt = tgt / tgt.sum() if tgt.sum() > 0 else tgt
                turn = M.turnover(current, tgt)
                cost = 2 * turn * value * bps / 1e4
            rows.append({"Strategy": ps.name, "Ann. return": ps.ann_return,
                         "Ann. vol": ps.ann_vol, "Sharpe": ps.sharpe, "Max DD": ps.max_dd,
                         "Turnover": turn, "Cost": cost})
        return pd.DataFrame(rows)

    def _render_trades(self, results) -> None:
        """Buy/sell trades to rebalance to the target (and, if available, the ORP)."""
        specs = [
            ("Portfolio Rebalancing Trades", "your target",
             "trade_recommendations_target", results.trade_recos),
        ]
        if results.trade_recos_orp is not None:
            specs.append(
                ("ORP Rebalancing Trades", "the optimal (ORP)",
                 "trade_recommendations_orp", results.trade_recos_orp)
            )

        for title, label, explain_key, df in specs:
            if df is None or df.empty:
                continue
            self.add_heading(title, explain=explain_key)
            traded = df[df["Action"] != "Hold"]
            if traded.empty:
                self.add_interpretation("Already on target — no trades needed.")
                continue

            buys = float(traded.loc[traded["TradeValue"] > 0, "TradeValue"].sum())
            sells = float(-traded.loc[traded["TradeValue"] < 0, "TradeValue"].sum())
            self.add_interpretation(
                f"To rebalance to {label}: buy ${buys:,.0f} and sell ${sells:,.0f} "
                f"across {len(traded)} position(s)."
            )
            self.add_chart(charts.trade_chart(df, f"Trades to reach {label}"), height=340)
            self.add_table(pd.DataFrame({
                "Ticker": df["Ticker"],
                "Current %": df["CurrentWeight"].map(_pct),
                "Target %": df["TargetWeight"].map(_pct),
                "Drift": df["Drift"].map(_pct),
                "Action": df["Action"],
                "Trade $": df["TradeValue"].map(_usd),
                "Shares": df["Shares"].map(_shares),
            }))
