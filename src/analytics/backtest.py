"""Inception-aware, cost-aware portfolio backtest engine.

This is the single source of truth for building a portfolio's daily value series.
It correctly handles assets with different inception dates (the "SPCX IPO'd last
week but is in a 5-year universe" problem) instead of truncating the whole
backtest to the newest asset's start date.

Two inception modes (user toggle):
- ``rescale`` (default): before an asset exists, hold the available assets at
  weights rescaled to sum to 1; when the asset's data begins, rebalance it in at
  its configured weight. Always fully invested in available assets.
- ``cash``: reserve each not-yet-available asset's target weight in a risk-free
  cash sleeve (growing at the risk-free rate) until its data begins, then deploy.

Rebalancing happens on asset-entry events plus an optional calendar schedule.
Transaction costs (bps of dollar traded) are deducted at each rebalance, and
every trade is recorded (the tax module consumes this trade log).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

# Calendar rebalance rules (pandas period-end aliases). "none" -> entry events only.
_FREQ_RULE = {
    "monthly": "ME",
    "quarterly": "QE",
    "semiannual": "2QE",
    "annual": "YE",
}


@dataclass
class BacktestResult:
    values: pd.Series                       # daily portfolio value
    weight_history: pd.DataFrame            # daily realized weights (+ "Cash" in cash mode)
    coverage: dict                          # ticker -> first-available Timestamp
    effective_start: Optional[pd.Timestamp] # date all weighted assets exist
    rebalance_dates: list                   # dates a rebalance occurred
    turnover_table: pd.DataFrame            # Date, Turnover (fraction of book traded)
    total_costs: float                      # total transaction costs ($)
    trades: pd.DataFrame                    # Date, Ticker, SharesDelta, Price, Value
    initial_holdings: pd.DataFrame          # Ticker, TargetWeight, PurchasePrice, Shares, Invested, RealizedWeight
    inception_mode: str = "rescale"


def build_backtest(
    prices: pd.DataFrame,
    weights: dict[str, float],
    capital: float,
    rf_annual: float = 0.0,
    inception_mode: str = "rescale",
    rebalance_frequency: str = "none",
    cost_bps: float = 0.0,
    start=None,
) -> BacktestResult:
    """Simulate a portfolio's daily value from prices + target weights."""
    cols = [t for t in weights if t in prices.columns]
    if not cols:
        raise ValueError("None of the weighted tickers have price data.")

    px_df = prices[cols].copy()
    if start is not None:
        px_df = px_df.loc[px_df.index >= pd.Timestamp(start)]

    # Per-asset inception = first date with real data in the window.
    coverage = {c: px_df[c].first_valid_index() for c in cols}
    cols = [c for c in cols if coverage[c] is not None]
    if not cols:
        raise ValueError("No price data for weighted tickers in window.")
    coverage = {c: coverage[c] for c in cols}

    first_any = min(coverage.values())
    px_df = px_df.loc[px_df.index >= first_any, cols]
    idx = px_df.index
    # Forward-fill gaps AFTER inception (a holiday/missing print), never before.
    P = px_df.ffill().values.astype(float)

    avail = np.zeros_like(P, dtype=bool)
    for j, c in enumerate(cols):
        avail[:, j] = idx >= coverage[c]
    # Guard against any residual NaN where "available" (shouldn't happen post-ffill).
    P = np.where(np.isnan(P), 0.0, P)

    w = np.array([float(weights[c]) for c in cols])
    n = len(cols)
    rf_daily = (1.0 + rf_annual) ** (1.0 / 252.0) - 1.0
    cost_rate = cost_bps / 1e4

    # ── Event dates: first day, each asset's entry, + optional calendar schedule ──
    idx_set = set(idx)
    events = {idx[0]}
    for c in cols:
        if coverage[c] in idx_set:
            events.add(coverage[c])
    rule = _FREQ_RULE.get(rebalance_frequency or "none")
    if rule:
        for mark in pd.Series(1, index=idx).resample(rule).last().index:
            sub = idx[idx <= mark]
            if len(sub):
                events.add(sub[-1])
    event_set = {d for d in events if d in idx_set}

    shares = np.zeros(n)
    cash = float(capital)
    total_costs = 0.0
    values = np.empty(len(idx))
    whist = np.zeros((len(idx), n))
    cashw = np.zeros(len(idx))
    trades: list[dict] = []
    turnover_rows: list[dict] = []
    rebalance_dates: list = []
    initial_holdings_rows: list[dict] = []

    for i, d in enumerate(idx):
        if i > 0:
            cash *= 1.0 + rf_daily
        price_row = P[i]
        avail_row = avail[i]
        asset_val = np.where(avail_row, shares * price_row, 0.0)
        total = asset_val.sum() + cash

        if d in event_set:
            # Target weights among currently-available assets.
            if inception_mode == "cash":
                tgt = np.where(avail_row, w, 0.0)
                cash_tgt = float(w[~avail_row].sum()) if (~avail_row).any() else 0.0
            else:  # rescale
                aw = np.where(avail_row, w, 0.0)
                s = aw.sum()
                tgt = aw / s if s > 0 else aw
                cash_tgt = 0.0

            valid = avail_row & (price_row > 0)
            # Estimate cost from provisional trade, then re-solve on value net of cost.
            est = np.divide(total * tgt, price_row, out=np.zeros(n), where=valid)
            traded = float((np.abs(est - shares) * np.where(valid, price_row, 0.0)).sum())
            cost = cost_rate * traded
            total_costs += cost
            value_after = total - cost
            new_shares = np.divide(value_after * tgt, price_row, out=np.zeros(n), where=valid)
            cash = value_after * cash_tgt

            for j, c in enumerate(cols):
                dsh = new_shares[j] - shares[j]
                if valid[j] and abs(dsh) > 1e-9:
                    trades.append(
                        {"Date": d, "Ticker": c, "SharesDelta": float(dsh),
                         "Price": float(price_row[j]), "Value": float(dsh * price_row[j])}
                    )
            if total > 0:
                turnover_rows.append({"Date": d, "Turnover": traded / total})
            rebalance_dates.append(d)
            shares = new_shares
            asset_val = np.where(avail_row, shares * price_row, 0.0)
            total = asset_val.sum() + cash

            if not initial_holdings_rows:  # capture the opening allocation
                for j, c in enumerate(cols):
                    if valid[j]:
                        invested = float(shares[j] * price_row[j])
                        initial_holdings_rows.append(
                            {"Ticker": c, "TargetWeight": float(weights[c]),
                             "PurchasePrice": float(price_row[j]), "Shares": float(shares[j]),
                             "Invested": invested}
                        )

        values[i] = total
        if total > 0:
            whist[i] = np.where(avail_row, shares * price_row, 0.0) / total
            cashw[i] = cash / total

    values_s = pd.Series(values, index=idx, name="Active")
    weight_history = pd.DataFrame(whist, index=idx, columns=cols)
    if inception_mode == "cash":
        weight_history["Cash"] = cashw

    effective_start = max(coverage.values()) if coverage else None

    holdings_df = pd.DataFrame(initial_holdings_rows)
    if not holdings_df.empty:
        tot_inv = holdings_df["Invested"].sum()
        holdings_df["RealizedWeight"] = holdings_df["Invested"] / tot_inv if tot_inv else 0.0

    return BacktestResult(
        values=values_s,
        weight_history=weight_history,
        coverage={c: coverage[c] for c in cols},
        effective_start=effective_start,
        rebalance_dates=rebalance_dates,
        turnover_table=pd.DataFrame(turnover_rows),
        total_costs=float(total_costs),
        trades=pd.DataFrame(trades),
        initial_holdings=holdings_df,
        inception_mode=inception_mode,
    )
