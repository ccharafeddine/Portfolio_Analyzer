"""
CAPM and multi-factor regression utilities.

Self-contained: no dependency on old analytics.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class RegressionResult:
    """Result of an OLS regression."""

    alpha: float
    beta: float | np.ndarray
    t_alpha: float
    t_beta: float | np.ndarray
    r_squared: float
    residual_std: float
    n_obs: int
    factor_names: list[str]

    # For multi-factor: betas as dict
    betas: dict[str, float] | None = None
    t_stats: dict[str, float] | None = None

    @property
    def ticker(self) -> str:
        """Convenience: first factor name is the ticker for CAPM results."""
        return self.factor_names[0] if self.factor_names else ""


def capm_regression(
    asset_returns: pd.Series,
    market_returns: pd.Series,
    rf_periodic: float = 0.0,
) -> RegressionResult:
    """
    Run CAPM regression: R_i - Rf = α + β(R_m - Rf) + ε

    Parameters
    ----------
    asset_returns : monthly/daily returns for the asset
    market_returns : monthly/daily returns for the market
    rf_periodic : periodic risk-free rate (same frequency as returns)

    Returns
    -------
    RegressionResult with alpha, beta, t-stats, R²
    """
    aligned = pd.concat(
        [asset_returns.rename("asset"), market_returns.rename("market")],
        axis=1,
    ).dropna()

    if len(aligned) < 5:
        return RegressionResult(
            alpha=np.nan, beta=np.nan, t_alpha=np.nan, t_beta=np.nan,
            r_squared=np.nan, residual_std=np.nan, n_obs=len(aligned),
            factor_names=["Market"],
        )

    y = aligned["asset"].values - rf_periodic
    x = aligned["market"].values - rf_periodic

    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

    n = len(y)
    y_hat = intercept + slope * x
    residuals = y - y_hat
    s2 = np.sum(residuals**2) / (n - 2)
    x_mean = x.mean()
    ss_x = np.sum((x - x_mean) ** 2)

    se_intercept = np.sqrt(s2 * (1.0 / n + x_mean**2 / ss_x))
    se_slope = np.sqrt(s2 / ss_x)

    t_alpha = intercept / se_intercept if se_intercept > 0 else np.nan
    t_beta = slope / se_slope if se_slope > 0 else np.nan

    return RegressionResult(
        alpha=float(intercept),
        beta=float(slope),
        t_alpha=float(t_alpha),
        t_beta=float(t_beta),
        r_squared=float(r_value**2),
        residual_std=float(np.sqrt(s2)),
        n_obs=n,
        factor_names=["Market"],
    )


def multi_factor_regression(
    asset_excess: pd.Series,
    factor_df: pd.DataFrame,
) -> RegressionResult:
    """
    Multi-factor OLS regression: R_i - Rf = α + Σ β_j F_j + ε

    Parameters
    ----------
    asset_excess : excess returns for one asset
    factor_df : DataFrame with factor columns (already excess of Rf)

    Returns
    -------
    RegressionResult with per-factor betas
    """
    aligned = pd.concat(
        [asset_excess.rename("Y"), factor_df], axis=1
    ).dropna()

    if len(aligned) < max(10, factor_df.shape[1] + 3):
        return RegressionResult(
            alpha=np.nan, beta=np.nan, t_alpha=np.nan, t_beta=np.nan,
            r_squared=np.nan, residual_std=np.nan, n_obs=len(aligned),
            factor_names=list(factor_df.columns),
        )

    y = aligned["Y"].values
    X = aligned.drop("Y", axis=1).values
    factor_names = list(factor_df.columns)
    n, k = X.shape

    # Add intercept
    X_aug = np.column_stack([np.ones(n), X])

    # OLS: β = (X'X)^-1 X'y
    try:
        XtX_inv = np.linalg.inv(X_aug.T @ X_aug)
    except np.linalg.LinAlgError:
        XtX_inv = np.linalg.pinv(X_aug.T @ X_aug)

    b = XtX_inv @ (X_aug.T @ y)
    y_hat = X_aug @ b
    residuals = y - y_hat
    s2 = np.sum(residuals**2) / (n - k - 1)
    se = np.sqrt(np.diag(XtX_inv) * s2)
    t_stats_all = b / se

    # R²
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    alpha = float(b[0])
    betas_arr = b[1:]
    t_alpha = float(t_stats_all[0])
    t_betas = t_stats_all[1:]

    betas_dict = {name: float(betas_arr[i]) for i, name in enumerate(factor_names)}
    t_dict = {name: float(t_betas[i]) for i, name in enumerate(factor_names)}

    return RegressionResult(
        alpha=alpha,
        beta=betas_arr,
        t_alpha=t_alpha,
        t_beta=t_betas,
        r_squared=float(r2),
        residual_std=float(np.sqrt(s2)),
        n_obs=n,
        factor_names=factor_names,
        betas=betas_dict,
        t_stats=t_dict,
    )


def run_capm_all(
    monthly_returns: pd.DataFrame,
    tickers: list[str],
    benchmark: str,
    rf_annual: float,
) -> list[RegressionResult]:
    """
    Run CAPM regressions for all tickers vs benchmark.

    Returns list of RegressionResults (one per ticker).
    """
    rf_m = (1 + rf_annual) ** (1 / 12) - 1
    results = []

    if benchmark not in monthly_returns.columns:
        return results

    bench_rets = monthly_returns[benchmark].dropna()

    for t in tickers:
        if t not in monthly_returns.columns or t == benchmark:
            continue

        aligned = pd.concat(
            [monthly_returns[t], bench_rets], axis=1, keys=[t, "mkt"]
        ).dropna()

        if aligned.empty:
            continue

        r = capm_regression(aligned[t], aligned["mkt"], rf_m)
        # Tag the ticker onto the result
        r.factor_names = [t]
        results.append(r)

    return results
