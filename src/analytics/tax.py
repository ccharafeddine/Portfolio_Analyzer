"""Tax-aware analysis (simple average-cost tier).

Uses the backtest's final positions + trade log and a per-ticker average cost
basis (user-supplied, or inferred from the opening purchase price) to estimate:
- unrealized gains/losses per holding,
- tax-loss-harvesting candidates (positions currently underwater),
- realized gains from rebalancing (split short/long-term by holding period),
- an estimated tax bill on those realized gains.

This is an estimate for planning, not tax advice. Lot-level accuracy (specific
lots, wash sales) is a later enhancement.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _avg_cost_fn(cost_basis: dict, purchase_prices: dict):
    def avg_cost(ticker: str) -> float:
        c = cost_basis.get(ticker, 0.0)
        if c and c > 0:
            return float(c)
        return float(purchase_prices.get(ticker, np.nan))
    return avg_cost


def build_tax_analysis(backtest, prices: pd.DataFrame, cost_basis: dict, tax_cfg):
    """Return (metrics: dict, detail: pd.DataFrame)."""
    wh = backtest.weight_history
    tickers = [c for c in wh.columns if c != "Cash"]
    final_val = float(backtest.values.iloc[-1])
    final_w = wh.iloc[-1]

    purchase_prices = {}
    if backtest.initial_holdings is not None and not backtest.initial_holdings.empty:
        purchase_prices = (
            backtest.initial_holdings.set_index("Ticker")["PurchasePrice"].to_dict()
        )
    avg_cost = _avg_cost_fn(cost_basis, purchase_prices)

    rows = []
    for t in tickers:
        w = float(final_w.get(t, 0.0))
        mv = final_val * w
        last_px = (
            float(prices[t].dropna().iloc[-1])
            if t in prices.columns and prices[t].notna().any()
            else np.nan
        )
        ac = avg_cost(t)
        shares = mv / last_px if last_px and last_px > 0 else 0.0
        cost = shares * ac if ac == ac else np.nan
        unreal = mv - cost if cost == cost else np.nan
        ret = (last_px / ac - 1) if (ac == ac and ac > 0) else np.nan
        rows.append(
            {
                "Ticker": t, "Shares": shares, "AvgCost": ac, "CurrentPrice": last_px,
                "MarketValue": mv, "CostBasis": cost, "UnrealizedGain": unreal, "Return": ret,
            }
        )
    detail = pd.DataFrame(rows)

    tot_unreal = float(detail["UnrealizedGain"].sum(skipna=True)) if not detail.empty else 0.0
    loss_mask = detail["UnrealizedGain"] < 0 if not detail.empty else pd.Series(dtype=bool)
    harvest = float(detail.loc[loss_mask, "UnrealizedGain"].sum()) if not detail.empty else 0.0
    n_harvest = int(loss_mask.sum()) if not detail.empty else 0

    # Realized gains from rebalancing sells, split by holding period.
    realized_st = realized_lt = 0.0
    trades = backtest.trades
    cov = backtest.coverage or {}
    if trades is not None and not trades.empty:
        for _, tr in trades.iterrows():
            if tr["SharesDelta"] >= 0:
                continue  # buys don't realize gains
            t = tr["Ticker"]
            ac = avg_cost(t)
            if ac != ac:
                continue
            shares_sold = -float(tr["SharesDelta"])
            gain = shares_sold * (float(tr["Price"]) - ac)
            acq = cov.get(t)
            held_days = (pd.Timestamp(tr["Date"]) - pd.Timestamp(acq)).days if acq is not None else 9999
            if held_days >= 365:
                realized_lt += gain
            else:
                realized_st += gain

    st_tax = max(realized_st, 0.0) * tax_cfg.short_term_rate
    lt_tax = max(realized_lt, 0.0) * tax_cfg.long_term_rate
    state_tax = max(realized_st + realized_lt, 0.0) * tax_cfg.state_rate
    est_tax = st_tax + lt_tax + state_tax

    metrics = {
        "unrealized_gain": tot_unreal,
        "harvest_potential": harvest,   # negative = harvestable losses
        "n_harvest": n_harvest,
        "realized_short": realized_st,
        "realized_long": realized_lt,
        "estimated_tax": est_tax,
        "cost_basis_source": "user" if cost_basis else "inferred from purchase price",
    }
    return metrics, detail
