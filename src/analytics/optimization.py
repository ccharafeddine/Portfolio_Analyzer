"""
Portfolio optimization — mean-variance, efficient frontier, max-Sharpe.

Self-contained: replaces old analytics.py optimization functions.
All math stays the same, but the API is cleaner and fully typed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


# ──────────────────────────────────────────────────────────────
# Core math
# ──────────────────────────────────────────────────────────────


def portfolio_return(weights: np.ndarray, mu: np.ndarray) -> float:
    """Expected portfolio return = w'μ."""
    return float(weights @ mu)


def portfolio_volatility(weights: np.ndarray, cov: np.ndarray) -> float:
    """Portfolio volatility = sqrt(w' Σ w), using annualized cov."""
    return float(np.sqrt(weights @ cov @ weights))


def sharpe_ratio(
    weights: np.ndarray,
    mu: np.ndarray,
    cov_monthly: np.ndarray,
    rf: float,
) -> float:
    """
    Annualized Sharpe ratio.

    Parameters
    ----------
    weights : (n,) array
    mu : (n,) annualized expected returns
    cov_monthly : (n, n) monthly covariance matrix
    rf : annual risk-free rate
    """
    cov_annual = cov_monthly * 12.0
    ret = portfolio_return(weights, mu)
    vol = portfolio_volatility(weights, cov_annual)
    if vol < 1e-12:
        return 0.0
    return float((ret - rf) / vol)


def _neg_sharpe(
    weights: np.ndarray,
    mu: np.ndarray,
    cov_monthly: np.ndarray,
    rf: float,
) -> float:
    """Negative Sharpe for minimization."""
    return -sharpe_ratio(weights, mu, cov_monthly, rf)


# ──────────────────────────────────────────────────────────────
# Max-Sharpe optimization
# ──────────────────────────────────────────────────────────────


def max_sharpe(
    mu: np.ndarray,
    cov_monthly: np.ndarray,
    rf: float = 0.04,
    bounds: tuple[float, float] = (0.0, 1.0),
    short_sales: bool = False,
) -> "minimize":
    """
    Find the max-Sharpe portfolio.

    Parameters
    ----------
    mu : (n,) annualized expected returns
    cov_monthly : (n, n) monthly covariance matrix
    rf : annual risk-free rate
    bounds : (lo, hi) weight bounds per asset
    short_sales : if True, lower bound is negative

    Returns
    -------
    scipy.optimize.OptimizeResult with .x = optimal weights
    """
    n = len(mu)
    lo = -bounds[1] if short_sales else bounds[0]
    hi = bounds[1]

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bnds = [(lo, hi)] * n
    x0 = np.ones(n) / n

    result = minimize(
        _neg_sharpe,
        x0,
        args=(mu, cov_monthly, rf),
        method="SLSQP",
        bounds=bnds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    return result


# ──────────────────────────────────────────────────────────────
# Efficient frontier
# ──────────────────────────────────────────────────────────────


def _min_vol_for_target(
    target_ret: float,
    mu: np.ndarray,
    cov_annual: np.ndarray,
    bounds: tuple[float, float],
    short_sales: bool,
) -> tuple[np.ndarray, float]:
    """Minimize volatility subject to a target return."""
    n = len(mu)
    lo = -bounds[1] if short_sales else bounds[0]
    hi = bounds[1]

    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        {"type": "eq", "fun": lambda w: portfolio_return(w, mu) - target_ret},
    ]

    result = minimize(
        lambda w: portfolio_volatility(w, cov_annual),
        np.ones(n) / n,
        method="SLSQP",
        bounds=[(lo, hi)] * n,
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-10},
    )
    vol = portfolio_volatility(result.x, cov_annual)
    return result.x, vol


def efficient_frontier(
    mu: np.ndarray,
    cov_monthly: np.ndarray,
    n_points: int = 50,
    bounds: tuple[float, float] = (0.0, 1.0),
    short_sales: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Trace the efficient frontier.

    Returns
    -------
    W : (n_points, n_assets) weight matrix
    R : (n_points,) expected returns
    V : (n_points,) expected volatilities
    """
    cov_annual = cov_monthly * 12.0
    n = len(mu)

    # Find feasible return range by solving min/max return
    lo_b = -bounds[1] if short_sales else bounds[0]
    hi_b = bounds[1]

    # Min return portfolio
    res_min = minimize(
        lambda w: portfolio_return(w, mu),
        np.ones(n) / n,
        method="SLSQP",
        bounds=[(lo_b, hi_b)] * n,
        constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}],
    )
    min_ret = portfolio_return(res_min.x, mu)

    # Max return portfolio
    res_max = minimize(
        lambda w: -portfolio_return(w, mu),
        np.ones(n) / n,
        method="SLSQP",
        bounds=[(lo_b, hi_b)] * n,
        constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}],
    )
    max_ret = portfolio_return(res_max.x, mu)

    # Pad slightly
    target_returns = np.linspace(min_ret + 0.001, max_ret - 0.001, n_points)

    W = np.zeros((n_points, n))
    R = np.zeros(n_points)
    V = np.zeros(n_points)

    for i, tr in enumerate(target_returns):
        w, vol = _min_vol_for_target(tr, mu, cov_annual, bounds, short_sales)
        W[i] = w
        R[i] = tr
        V[i] = vol

    return W, R, V


# ──────────────────────────────────────────────────────────────
# Global Minimum Variance
# ──────────────────────────────────────────────────────────────


def min_variance_portfolio(
    cov_monthly: np.ndarray,
    bounds: tuple[float, float] = (0.0, 1.0),
    short_sales: bool = False,
) -> np.ndarray:
    """Find the global minimum variance portfolio."""
    cov_annual = cov_monthly * 12.0
    n = cov_annual.shape[0]
    lo = -bounds[1] if short_sales else bounds[0]
    hi = bounds[1]

    result = minimize(
        lambda w: portfolio_volatility(w, cov_annual),
        np.ones(n) / n,
        method="SLSQP",
        bounds=[(lo, hi)] * n,
        constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}],
        options={"maxiter": 500},
    )
    return result.x
