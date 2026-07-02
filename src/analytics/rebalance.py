"""
Rebalancing analysis: weight drift, periodic rebalancing backtest, turnover,
and concrete buy/sell trade recommendations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def trade_recommendations(
    current_weights: pd.Series,
    target_weights: dict[str, float],
    total_value: float,
    latest_prices: pd.Series,
    band: float = 0.0,
) -> pd.DataFrame:
    """Concrete trades to move from current (drifted) weights to a target allocation.

    Parameters
    ----------
    current_weights : realized weight per ticker now (sums to ~1).
    target_weights : desired weight per ticker (normalized internally).
    total_value : current portfolio value ($).
    latest_prices : latest price per ticker (for share counts).
    band : no-trade band — |drift| below this is left as "Hold" (0 = trade everything).

    Returns numeric columns: Ticker, CurrentWeight, TargetWeight, Drift,
    CurrentValue, TargetValue, TradeValue (+buy/-sell), Shares, Action —
    sorted by trade size (largest first).
    """
    tgt_total = sum(v for v in target_weights.values() if v)
    tgt = {k: (v / tgt_total if tgt_total else 0.0) for k, v in target_weights.items()}

    tickers = sorted(set(current_weights.index) | set(tgt))
    rows = []
    for t in tickers:
        cur_w = float(current_weights.get(t, 0.0) or 0.0)
        tgt_w = float(tgt.get(t, 0.0) or 0.0)
        drift = cur_w - tgt_w
        cur_val = cur_w * total_value
        tgt_val = tgt_w * total_value
        trade_val = tgt_val - cur_val  # +buy / -sell

        if abs(drift) < band:
            action, trade_val = "Hold", 0.0
        elif trade_val > 0:
            action = "Buy"
        elif trade_val < 0:
            action = "Sell"
        else:
            action = "Hold"

        price = float(latest_prices.get(t, np.nan)) if t in latest_prices.index else np.nan
        shares = (trade_val / price) if (price and price == price and price > 0) else np.nan

        rows.append({
            "Ticker": t,
            "CurrentWeight": cur_w,
            "TargetWeight": tgt_w,
            "Drift": drift,
            "CurrentValue": cur_val,
            "TargetValue": tgt_val,
            "TradeValue": trade_val,
            "Shares": shares,
            "Action": action,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.reindex(
        df["TradeValue"].abs().sort_values(ascending=False).index
    ).reset_index(drop=True)


def drift_from_target(
    prices: pd.DataFrame,
    target_weights: dict[str, float],
    purchase_date: pd.Timestamp | str,
) -> pd.DataFrame:
    """
    Compute daily weight drift from target allocation.

    Returns a DataFrame with tickers as columns and dates as index,
    where each value is the portfolio weight of that ticker on that day.
    Weights sum to 1.0 each row.
    """
    purchase_date = pd.Timestamp(purchase_date)
    tickers = [t for t in target_weights if t in prices.columns]
    if not tickers:
        return pd.DataFrame()

    sub = prices[tickers].loc[prices.index >= purchase_date].dropna(how="any")
    if sub.empty:
        return pd.DataFrame()

    initial_prices = sub.iloc[0]
    shares = {}
    for t in tickers:
        shares[t] = target_weights[t] / initial_prices[t]

    # Daily portfolio values per asset
    values = pd.DataFrame({t: sub[t] * shares[t] for t in tickers})
    total = values.sum(axis=1)
    drift = values.div(total, axis=0)
    return drift


def rebalanced_backtest(
    prices: pd.DataFrame,
    target_weights: dict[str, float],
    capital: float,
    frequency: str = "quarterly",
) -> pd.Series:
    """
    Simulate a buy-and-rebalance strategy.

    Parameters
    ----------
    prices : daily price data
    target_weights : {ticker: weight}
    capital : initial capital
    frequency : 'monthly', 'quarterly', or 'annual'

    Returns
    -------
    pd.Series of daily portfolio values
    """
    tickers = [t for t in target_weights if t in prices.columns]
    if not tickers:
        return pd.Series(dtype=float)

    sub = prices[tickers].dropna(how="any")
    if sub.empty:
        return pd.Series(dtype=float)

    freq_map = {"monthly": "ME", "quarterly": "QE", "annual": "YE"}
    rule = freq_map.get(frequency, "QE")

    # Identify rebalance dates
    rebal_dates = sub.resample(rule).last().index
    rebal_set = set(rebal_dates)

    # Initial allocation
    w = np.array([target_weights[t] for t in tickers])
    initial_prices = sub.iloc[0].values
    shares = (capital * w) / initial_prices
    portfolio_value = float(np.sum(shares * initial_prices))

    values = []
    for dt, row in sub.iterrows():
        pv = float(np.sum(shares * row.values))

        if dt in rebal_set and dt != sub.index[0]:
            # Rebalance: redistribute current value to target weights
            shares = (pv * w) / row.values

        values.append(pv)

    result = pd.Series(values, index=sub.index, name="Rebalanced")
    return result


def compute_turnover(
    weight_drift: pd.DataFrame,
    frequency: str = "quarterly",
) -> pd.DataFrame:
    """
    Compute periodic turnover from weight drift.

    Turnover = sum of absolute weight changes at each rebalance point.

    Returns DataFrame with columns: Date, Turnover
    """
    if weight_drift.empty:
        return pd.DataFrame(columns=["Date", "Turnover"])

    freq_map = {"monthly": "ME", "quarterly": "QE", "annual": "YE"}
    rule = freq_map.get(frequency, "QE")

    rebal_dates = weight_drift.resample(rule).last().index
    target = weight_drift.iloc[0].values

    rows = []
    for dt in rebal_dates:
        if dt not in weight_drift.index:
            continue
        current = weight_drift.loc[dt].values
        turnover = float(np.sum(np.abs(current - target)))
        rows.append({"Date": dt, "Turnover": turnover})

    return pd.DataFrame(rows)
