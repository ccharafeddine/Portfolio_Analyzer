"""
black_litterman.py

Black-Litterman implementation for the Portfolio Analyzer.

- Computes market-cap equilibrium returns (pi)
- Builds P, Q, and Omega from absolute/relative views
- Computes posterior BL mean & covariance
- Optimizes BL portfolio using existing Markowitz optimizer (max_sharpe)
"""

import numpy as np
import pandas as pd
import requests
from typing import Dict, List, Tuple

from analytics import max_sharpe  # already in your project


# -----------------------------------------------------------------------------
# 1. Market Cap Weights (FMP API)
# -----------------------------------------------------------------------------
def get_market_cap_weights_fmp(tickers: List[str]) -> pd.Series:
    """
    Fetch market caps from FMP and return normalized market-cap weights.

    If any ticker fails, its market cap is set to 0.
    """
    key = _get_fmp_api_key()
    base = "https://financialmodelingprep.com/api/v3/market-capitalization/{}?apikey={}"

    caps = {}
    for t in tickers:
        url = base.format(t, key)
        try:
            r = requests.get(url, timeout=10)
            data = r.json()
            if isinstance(data, list) and len(data) > 0 and "marketCap" in data[0]:
                caps[t] = float(data[0]["marketCap"])
            else:
                caps[t] = 0.0
        except Exception:
            caps[t] = 0.0

    s = pd.Series(caps, dtype=float)
    total = s.sum()
    if total <= 0:
        # fallback to equal weights
        return pd.Series(1 / len(tickers), index=tickers)
    return s / total


def _get_fmp_api_key() -> str:
    """Helper to get FMP key from env or Streamlit."""
    import os

    key = os.getenv("FMP_API_KEY")
    if key:
        return key

    # Streamlit secrets fallback
    try:
        import streamlit as st
        if "FMP_API_KEY" in st.secrets:
            return st.secrets["FMP_API_KEY"]
    except Exception:
        pass

    raise RuntimeError("FMP_API_KEY not found.")


# -----------------------------------------------------------------------------
# 2. Implied Equilibrium Returns (pi)
# -----------------------------------------------------------------------------
def compute_black_litterman_prior(
    prices: pd.DataFrame,
    market_caps: pd.Series,
    rf_annual: float = 0.04,
    periods_per_year: int = 252
) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Compute equilibrium expected returns using the CAPM / BL equilibrium formula.

    pi = δ * Σ * w_mkt

    Where:
    - Σ is the covariance matrix of *excess returns*
    - w_mkt are market-cap weights
    - δ is risk aversion parameter

    Returns:
        pi (Series) - implied equilibrium excess returns
        cov (DataFrame) - historical covariance matrix of returns
    """

    # Daily returns
    rets = prices.pct_change().dropna(how="all")

    # Monthly covariance (as expected in your analytics pipeline)
    cov = rets.cov() * periods_per_year  # annualize

    # Compute market portfolio variance
    w = market_caps.reindex(cov.index).fillna(0).values
    portfolio_var = float(w @ cov.values @ w)
    if portfolio_var <= 0:
        portfolio_var = 1e-8

    # Risk aversion δ ≈ (E[R_mkt] − rf) / Var_mkt
    # Use historical market return proxy
    mean_rets = rets.mean() * periods_per_year
    mkt_return = float((w * mean_rets).sum())
    delta = (mkt_return - rf_annual) / portfolio_var

    # Implied equilibrium returns
    pi = pd.Series(delta * cov.values.dot(w), index=cov.index)
    return pi, cov


# -----------------------------------------------------------------------------
# 3. Build P, Q, Omega from views
# -----------------------------------------------------------------------------
def build_P_Q_Omega(
    tickers: List[str],
    views: List[Dict],
    cov: pd.DataFrame,
    tau: float = 0.05
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build generalized P, Q, and Omega matrices for Black-Litterman.

    Supported view formats:
      - Absolute view:
            {"type":"absolute", "asset":"AAPL", "q":0.08, "confidence":"medium"}

      - Relative view:
            {"type":"relative", "asset_long":"AAPL", "asset_short":"MSFT",
             "q":0.03, "confidence":"high"}

    Confidence levels -> scaled Omega:
        low     => large uncertainty
        medium  => moderate
        high    => small uncertainty
    """

    if len(views) == 0:
        return None, None, None

    n = len(tickers)
    k = len(views)

    P = np.zeros((k, n))
    Q = np.zeros(k)
    Omega = np.zeros((k, k))

    # Simple mapping for confidence -> scaling
    conf_scale = {"low": 10.0, "medium": 1.0, "high": 0.1}

    for i, v in enumerate(views):
        vtype = v.get("type", "").lower()
        conf = v.get("confidence", "medium").lower()
        scale = conf_scale.get(conf, 1.0)

        # ---------------------------------------------------------------------
        # Absolute view: r_asset = q
        # ---------------------------------------------------------------------
        if vtype == "absolute":
            asset = v["asset"]
            if asset not in tickers:
                raise ValueError(f"Asset {asset} not in universe.")

            P[i, tickers.index(asset)] = 1.0
            Q[i] = float(v["q"])

        # ---------------------------------------------------------------------
        # Relative view: r_long − r_short = q
        # ---------------------------------------------------------------------
        elif vtype == "relative":
            a = v["asset_long"]
            b = v["asset_short"]
            if a not in tickers or b not in tickers:
                raise ValueError(f"Relative view assets {a}/{b} not in universe.")

            P[i, tickers.index(a)] = 1.0
            P[i, tickers.index(b)] = -1.0
            Q[i] = float(v["q"])

        else:
            raise ValueError(f"Invalid view type: {vtype}")

        # Omega diagonal
        # Standard BL heuristic: Omega = diag(P Σ P') * tau * scale
        row = P[i, :].reshape(1, -1)
        variance = float(row @ cov.values @ row.T)
        Omega[i, i] = max(variance * tau * scale, 1e-12)

    return P, Q, Omega


# -----------------------------------------------------------------------------
# 4. Posterior BL Mean and Covariance
# -----------------------------------------------------------------------------
def compute_bl_posterior(
    pi: pd.Series,
    cov: pd.DataFrame,
    P: np.ndarray,
    Q: np.ndarray,
    Omega: np.ndarray,
    tau: float = 0.05
) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Compute Black-Litterman posterior returns and posterior covariance.

    If P/Q/Omega are None (no views), returns the prior (pi, cov).
    """

    if P is None or Q is None or Omega is None:
        # No views = BL collapses to equilibrium prior
        return pi, cov

    Sigma = cov.values
    tickers = list(pi.index)

    TauSigmaInv = np.linalg.inv(tau * Sigma)
    OmegaInv = np.linalg.inv(Omega)

    A = TauSigmaInv + P.T @ OmegaInv @ P
    b = TauSigmaInv @ pi.values + P.T @ OmegaInv @ Q

    mu_post = np.linalg.solve(A, b)
    cov_post = np.linalg.inv(A)

    mu_series = pd.Series(mu_post, index=tickers)
    cov_df = pd.DataFrame(cov_post, index=tickers, columns=tickers)

    return mu_series, cov_df


# -----------------------------------------------------------------------------
# 5. Optimize using existing Markowitz (max_sharpe)
# -----------------------------------------------------------------------------
def optimize_bl_weights(
    mu_bl: pd.Series,
    cov_bl: pd.DataFrame,
    rf_annual: float,
    bounds: Tuple[float, float],
    short_sales: bool
) -> pd.Series:
    """
    Optimize BL portfolio using your existing max_sharpe solver.
    """

    mu = mu_bl.values
    # Note: optimizer expects monthly covariance matrix (cov_m)
    cov_m = cov_bl.values / 12.0  # convert annual → monthly

    res = max_sharpe(
        mu_a=mu,
        cov_m=cov_m,
        rf=rf_annual,
        bounds=bounds,
        short_sales=short_sales
    )

    w = pd.Series(res.x, index=mu_bl.index, dtype=float)
    return w