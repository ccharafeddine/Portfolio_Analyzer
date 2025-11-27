
import numpy as np
import pandas as pd
from typing import Tuple
from scipy.optimize import minimize
import statsmodels.api as sm

def summarize_returns(rets: pd.DataFrame) -> pd.DataFrame:
    """Annualize mean/vol using monthly returns."""
    mu_m = rets.mean()
    cov_m = rets.cov()
    mu_a = (1 + mu_m)**12 - 1
    vol_a = np.sqrt(np.diag(cov_m) * 12)
    sharpe_dummy = mu_a / pd.Series(vol_a, index=rets.columns).replace(0, np.nan)
    return pd.DataFrame({"ann_mean": mu_a, "ann_vol": vol_a, "mean_monthly": mu_m, "sharpe_no_rf": sharpe_dummy})

def sharpe_ratio(weights: np.ndarray, mu_a: np.ndarray, cov_m: np.ndarray, rf: float) -> float:
    """Annualized Sharpe using monthly cov matrix; rf is annual rate."""
    cov_a = cov_m * 12.0
    port_mean = float(weights @ mu_a)
    port_vol = float(np.sqrt(weights @ cov_a @ weights))
    return (port_mean - rf) / port_vol if port_vol > 0 else -np.inf

def max_sharpe(mu_a: np.ndarray, cov_m: np.ndarray, rf: float, bounds: Tuple[float, float], short_sales: bool = False):
    """Solve for max Sharpe weights under sum-to-1 and bounds."""
    n = len(mu_a)
    bnds = tuple((bounds[0] if short_sales else 0.0, bounds[1]) for _ in range(n))
    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    x0 = np.repeat(1.0 / n, n)
    def neg_sharpe(w): return -sharpe_ratio(w, mu_a, cov_m, rf)
    res = minimize(neg_sharpe, x0, method="SLSQP", bounds=bnds, constraints=cons, options={"maxiter": 300})
    return res

def efficient_frontier(mu_a: np.ndarray, cov_m: np.ndarray, n_points: int, bounds: Tuple[float, float], short_sales: bool = False):
    """Trace the minimum-variance portfolios for target returns across range."""
    n = len(mu_a)
    target_returns = np.linspace(mu_a.min(), mu_a.max(), n_points)
    cov_a = cov_m * 12.0
    W, R, V = [], [], []
    for tr in target_returns:
        cons = (
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            {"type": "eq", "fun": lambda w, tr=tr: float(w @ mu_a - tr)},
        )
        bnds = tuple((bounds[0] if short_sales else 0.0, bounds[1]) for _ in range(n))
        x0 = np.repeat(1.0 / n, n)
        obj = lambda w: float(w @ cov_a @ w)
        res = minimize(obj, x0, method="SLSQP", bounds=bnds, constraints=cons)
        if res.success:
            w = res.x
            W.append(w)
            R.append(float(w @ mu_a))
            V.append(float(np.sqrt(w @ cov_a @ w)))
    return np.array(W), np.array(R), np.array(V)

def capm_regression(asset_rets_m: pd.Series, market_rets_m: pd.Series, rf_m: float = 0.0):
    """Run CAPM monthly: (R_i - R_f) = alpha + beta*(R_m - R_f) + eps."""
    ex_i = asset_rets_m - rf_m
    ex_m = market_rets_m - rf_m
    X = sm.add_constant(ex_m.values)
    y = ex_i.values
    model = sm.OLS(y, X).fit()
    alpha, beta = float(model.params[0]), float(model.params[1])
    return {
        "alpha": alpha, "beta": beta,
        "t_alpha": float(model.tvalues[0]), "t_beta": float(model.tvalues[1]),
        "r2": float(model.rsquared)
    }
