"""
Sector and factor exposure analysis.

- Sector weights via yfinance metadata
- Return-based factor tilts (beta, size proxy, momentum, quality)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    yf = None


def get_sector_weights(holdings: pd.DataFrame) -> pd.DataFrame:
    """
    Map each holding to its sector via yfinance and compute sector weights.

    Parameters
    ----------
    holdings : DataFrame with 'Ticker' and 'RealizedWeight' columns

    Returns
    -------
    DataFrame with columns: Sector, Weight
    """
    if yf is None:
        return pd.DataFrame(columns=["Sector", "Weight"])

    sector_map = {}
    for ticker in holdings["Ticker"].unique():
        try:
            info = yf.Ticker(ticker).info
            sector_map[ticker] = info.get("sector", "Unknown")
        except Exception:
            sector_map[ticker] = "Unknown"

    df = holdings[["Ticker", "RealizedWeight"]].copy()
    df["Sector"] = df["Ticker"].map(sector_map)

    sector_weights = df.groupby("Sector")["RealizedWeight"].sum().reset_index()
    sector_weights.columns = ["Sector", "Weight"]
    sector_weights = sector_weights.sort_values("Weight", ascending=False)
    return sector_weights


def get_factor_tilts(
    monthly_returns: pd.DataFrame,
    market_returns: pd.Series,
    window: int = 36,
) -> pd.DataFrame:
    """
    Compute return-based factor tilt proxies for each asset.

    Factors computed:
    - Beta: rolling covariance with market / market variance
    - Size: volatility rank (higher vol = smaller cap proxy)
    - Momentum: trailing 12-1 month return
    - Quality: trailing Sharpe ratio proxy

    Parameters
    ----------
    monthly_returns : asset monthly returns
    market_returns : market/benchmark monthly returns
    window : lookback window in months

    Returns
    -------
    DataFrame with columns: Asset, Beta, Size, Momentum, Quality
    """
    rows = []
    for col in monthly_returns.columns:
        asset = monthly_returns[col].dropna()
        mkt = market_returns.reindex(asset.index).dropna()
        common = asset.index.intersection(mkt.index)

        if len(common) < max(13, window // 2):
            rows.append({
                "Asset": col,
                "Beta": np.nan,
                "Size": np.nan,
                "Momentum": np.nan,
                "Quality": np.nan,
            })
            continue

        asset_c = asset.loc[common]
        mkt_c = mkt.loc[common]

        # Beta
        cov_val = asset_c.cov(mkt_c)
        var_val = mkt_c.var()
        beta = cov_val / var_val if var_val > 1e-12 else np.nan

        # Size proxy: annualized vol (higher vol = more small-cap-like)
        size_proxy = float(asset_c.std() * np.sqrt(12))

        # Momentum: 12-1 month return (skip most recent month)
        if len(asset_c) >= 13:
            mom = float((1 + asset_c.iloc[-13:-1]).prod() - 1)
        else:
            mom = np.nan

        # Quality proxy: trailing Sharpe
        ann_ret = float((1 + asset_c.mean()) ** 12 - 1)
        ann_vol = float(asset_c.std() * np.sqrt(12))
        quality = ann_ret / ann_vol if ann_vol > 1e-12 else np.nan

        rows.append({
            "Asset": col,
            "Beta": round(beta, 3) if not np.isnan(beta) else np.nan,
            "Size": round(size_proxy, 3),
            "Momentum": round(mom, 3) if not np.isnan(mom) else np.nan,
            "Quality": round(quality, 3) if not np.isnan(quality) else np.nan,
        })

    return pd.DataFrame(rows)


def effective_n_sectors(sector_weights: pd.DataFrame) -> float:
    """
    Effective number of sectors = 1 / HHI of sector weights.

    Parameters
    ----------
    sector_weights : DataFrame with 'Weight' column

    Returns
    -------
    float: effective number of sectors
    """
    if sector_weights.empty or "Weight" not in sector_weights.columns:
        return 0.0

    w = sector_weights["Weight"].values
    total = w.sum()
    if total < 1e-12:
        return 0.0

    w_norm = w / total
    hhi = float(np.sum(w_norm ** 2))
    if hhi < 1e-12:
        return 0.0
    return 1.0 / hhi
