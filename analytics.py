import os
from typing import Tuple, Dict, Iterable, Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import statsmodels.api as sm


# ----------------------------------------------------------------------
# Mean/variance/CAPM utilities (existing behaviour, kept compatible)
# ----------------------------------------------------------------------
def summarize_returns(rets: pd.DataFrame) -> pd.DataFrame:
    """Annualize mean/vol using monthly returns."""
    mu_m = rets.mean()
    cov_m = rets.cov()
    mu_a = (1 + mu_m) ** 12 - 1
    vol_a = np.sqrt(np.diag(cov_m) * 12)
    sharpe_dummy = mu_a / pd.Series(vol_a, index=rets.columns).replace(0, np.nan)
    return pd.DataFrame(
        {
            "ann_mean": mu_a,
            "ann_vol": vol_a,
            "mean_monthly": mu_m,
            "sharpe_no_rf": sharpe_dummy,
        }
    )


def sharpe_ratio(
    weights: np.ndarray,
    mu_a: np.ndarray,
    cov_m: np.ndarray,
    rf: float,
) -> float:
    """Annualized Sharpe using monthly covariance matrix, rf annual."""
    cov_a = cov_m * 12.0
    port_mean = float(weights @ mu_a)
    port_vol = float(np.sqrt(weights @ cov_a @ weights))
    return (port_mean - rf) / port_vol if port_vol > 0 else -np.inf


def max_sharpe(
    mu_a: np.ndarray,
    cov_m: np.ndarray,
    rf: float,
    bounds: Tuple[float, float],
    short_sales: bool = False,
):
    """Solve for max-Sharpe weights under sum-to-1 and box constraints."""
    n = len(mu_a)
    bnds = tuple((bounds[0] if short_sales else 0.0, bounds[1]) for _ in range(n))
    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    x0 = np.repeat(1.0 / n, n)

    def neg_sharpe(w: np.ndarray) -> float:
        return -sharpe_ratio(w, mu_a, cov_m, rf)

    res = minimize(
        neg_sharpe,
        x0,
        method="SLSQP",
        bounds=bnds,
        constraints=cons,
        options={"maxiter": 300},
    )
    return res


def efficient_frontier(
    mu_a: np.ndarray,
    cov_m: np.ndarray,
    n_points: int,
    bounds: Tuple[float, float],
    short_sales: bool = False,
):
    """Trace the minimum-variance portfolios for a grid of target returns."""
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


def capm_regression(
    asset_rets_m: pd.Series,
    market_rets_m: pd.Series,
    rf_m: float = 0.0,
) -> Dict[str, float]:
    """Run CAPM monthly: (R_i − R_f) = alpha + beta (R_m − R_f) + eps."""
    ex_i = asset_rets_m - rf_m
    ex_m = market_rets_m - rf_m
    X = sm.add_constant(ex_m.values)
    y = ex_i.values
    model = sm.OLS(y, X).fit()
    alpha, beta = float(model.params[0]), float(model.params[1])
    return {
        "alpha": alpha,
        "beta": beta,
        "t_alpha": float(model.tvalues[0]),
        "t_beta": float(model.tvalues[1]),
        "r2": float(model.rsquared),
    }


# ----------------------------------------------------------------------
# NEW: Drawdown & tail-risk helpers
# ----------------------------------------------------------------------
def compute_drawdown(series: pd.Series) -> pd.Series:
    """Compute percentage drawdown series from a value series.

    Returns a Series in decimal form (-0.20 = -20%).
    """
    s = series.sort_index().astype(float).dropna()
    if s.empty:
        return pd.Series(dtype="float64", name=getattr(series, "name", None))

    running_max = s.cummax()
    dd = s / running_max - 1.0
    dd.name = getattr(series, "name", None)
    return dd


def summarize_drawdown(series: pd.Series) -> Dict[str, Optional[float]]:
    """Max drawdown + peak / trough / recovery dates & time-to-recover."""
    s = series.sort_index().astype(float).dropna()
    if s.empty:
        return {
            "max_drawdown": np.nan,
            "dd_start_date": np.nan,
            "dd_trough_date": np.nan,
            "dd_recovery_date": np.nan,
            "dd_time_to_recover_days": np.nan,
        }

    running_max = s.cummax()
    dd = s / running_max - 1.0

    max_dd = float(dd.min())
    trough_date = dd.idxmin()

    window_running_max = running_max.loc[:trough_date]
    peak_value = float(window_running_max.max())
    peak_date = window_running_max.idxmax()

    post_trough = s.loc[trough_date:]
    recovered = post_trough[post_trough >= peak_value]
    if not recovered.empty:
        recovery_date = recovered.index[0]
        ttr_days = float((recovery_date - peak_date).days)
        recovery_date_val: Optional[pd.Timestamp] = recovery_date
    else:
        recovery_date_val = None
        ttr_days = np.nan

    return {
        "max_drawdown": max_dd,
        "dd_start_date": peak_date,
        "dd_trough_date": trough_date,
        "dd_recovery_date": recovery_date_val,
        "dd_time_to_recover_days": ttr_days,
    }


def _var_cvar_from_returns(
    rets: pd.Series,
    alpha: float = 0.95,
) -> Tuple[float, float]:
    """Historical one-period VaR/CVaR for a simple-return series."""
    r = rets.dropna().astype(float)
    if r.empty:
        return np.nan, np.nan

    p = 1.0 - float(alpha)
    var = float(r.quantile(p))
    tail = r[r <= var]
    cvar = float(tail.mean()) if not tail.empty else np.nan
    return var, cvar


def downside_deviation(
    rets: pd.Series,
    mar: float = 0.0,
    periods_per_year: int = 252,
) -> Tuple[float, float]:
    """Downside deviation vs MAR, per-period and annualized."""
    r = rets.dropna().astype(float)
    if r.empty:
        return np.nan, np.nan

    diff = r - mar
    downside = diff[diff < 0.0]
    if downside.empty:
        return 0.0, 0.0

    dd_periodic = float(downside.std())
    dd_annual = float(dd_periodic * np.sqrt(periods_per_year))
    return dd_periodic, dd_annual


def _load_portfolio_value_series(
    outdir: str,
    mapping: Dict[str, Tuple[str, Iterable[str]]],
) -> Dict[str, pd.Series]:
    """Internal helper: load portfolio value series from CSVs."""
    out: Dict[str, pd.Series] = {}
    for name, (fname, preferred_cols) in mapping.items():
        path = os.path.join(outdir, fname)
        if not os.path.exists(path):
            continue

        df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")

        col_to_use: Optional[str] = None
        for col in preferred_cols:
            if col in df.columns:
                col_to_use = col
                break

        if col_to_use is None:
            num_cols = df.select_dtypes(include=["number"]).columns
            if len(num_cols) > 0:
                col_to_use = num_cols[0]
            elif len(df.columns) > 0:
                col_to_use = df.columns[0]
            else:
                continue

        s = df[col_to_use].astype(float).sort_index()
        s.name = name
        s = s.dropna()
        if not s.empty:
            out[name] = s

    return out


def compute_all_portfolio_drawdown_and_tail_risk(
    outdir: str = "outputs",
    rf_annual: float = 0.02,  # kept for future use / report text
    alpha_levels: Iterable[float] = (0.95, 0.99),
    periods_per_year: int = 252,
) -> None:
    """Compute drawdown & tail-risk metrics for key portfolios.

    Saves two CSVs in ``outdir``:

      * drawdown_series.csv
      * drawdown_tail_metrics.csv
    """
    mapping: Dict[str, Tuple[str, Iterable[str]]] = {
        "Active": ("active_portfolio_value.csv", ["PortfolioValue", "Active", "Value"]),
        "Passive": ("passive_portfolio_value.csv", ["Passive", "Value"]),
        "ORP": ("orp_value_realized.csv", ["ORP_Value", "Value"]),
        "Complete": (
            "complete_portfolio_value.csv",
            ["complete_portfolio_value", "Complete", "Value"],
        ),
    }

    portfolios = _load_portfolio_value_series(outdir, mapping)
    if not portfolios:
        print("[drawdown/tail] No portfolio value series found; skipping metrics.")
        return

    all_index: Optional[pd.DatetimeIndex] = None
    for s in portfolios.values():
        all_index = s.index if all_index is None else all_index.union(s.index)
    assert all_index is not None

    dd_df = pd.DataFrame(index=all_index)
    rows = []

    for name, series in portfolios.items():
        dd = compute_drawdown(series)
        dd_df[name] = dd

        daily_rets = series.sort_index().pct_change().dropna()
        dd_stats = summarize_drawdown(series)

        dd_periodic, dd_annual = downside_deviation(
            daily_rets,
            mar=0.0,
            periods_per_year=periods_per_year,
        )

        var_results: Dict[str, float] = {}
        cvar_results: Dict[str, float] = {}
        for a in alpha_levels:
            var, cvar = _var_cvar_from_returns(daily_rets, alpha=a)
            suffix = str(int(round(a * 100)))
            var_results[f"VaR_{suffix}"] = var
            cvar_results[f"CVaR_{suffix}"] = cvar

        row = {
            "Portfolio": name,
            "max_drawdown": dd_stats["max_drawdown"],
            "dd_start_date": dd_stats["dd_start_date"],
            "dd_trough_date": dd_stats["dd_trough_date"],
            "dd_recovery_date": dd_stats["dd_recovery_date"],
            "dd_time_to_recover_days": dd_stats["dd_time_to_recover_days"],
            "downside_dev_daily": dd_periodic,
            "downside_dev_annual": dd_annual,
        }
        row.update(var_results)
        row.update(cvar_results)
        rows.append(row)

    dd_df.sort_index().to_csv(os.path.join(outdir, "drawdown_series.csv"), index_label="Date")

    if rows:
        metrics_df = pd.DataFrame(rows).set_index("Portfolio")
        metrics_df.to_csv(os.path.join(outdir, "drawdown_tail_metrics.csv"))
    else:
        print("[drawdown/tail] No metrics rows; nothing to save.")
