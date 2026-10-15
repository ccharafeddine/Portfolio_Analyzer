"""Fama-French style factor regressions.

Regress each asset's (and the portfolio's) daily excess returns on a factor set to get
true factor loadings (betas), alpha, and R² — replacing the old return-based proxies.
Output columns match ``charts.factor_loadings_chart``: ``Asset``, ``const_coef`` (alpha),
one ``<factor>_coef`` per factor, matching ``*_t`` t-statistics
(OLS, conventional SEs), ``R2`` and ``N``.
"""

from __future__ import annotations

import pandas as pd

import logging

log = logging.getLogger(__name__)

# Factor columns per model (must exist in the fetched Fama-French frame).
FACTOR_SETS: dict[str, list[str]] = {
    "FF3": ["Mkt-RF", "SMB", "HML"],
    "Carhart 4-Factor": ["Mkt-RF", "SMB", "HML", "MOM"],
    "FF5": ["Mkt-RF", "SMB", "HML", "RMW", "CMA"],
}

_MIN_OBS = 40


def run_factor_model(
    daily_returns: pd.DataFrame,
    ff: pd.DataFrame,
    factor_cols: list[str],
    port_returns: pd.Series | None = None,
) -> pd.DataFrame:
    """Loadings table (one row per asset, plus a leading Portfolio row if given)."""
    import statsmodels.api as sm

    if ff is None or ff.empty or not all(c in ff.columns for c in factor_cols):
        return pd.DataFrame()
    if "RF" not in ff.columns:
        return pd.DataFrame()

    series: dict[str, pd.Series] = {}
    if port_returns is not None and not port_returns.empty:
        series["Portfolio"] = port_returns
    for c in daily_returns.columns:
        series[c] = daily_returns[c]

    rows = []
    for name, r in series.items():
        data = pd.concat(
            [r.rename("y"), ff[factor_cols], ff["RF"].rename("RF")], axis=1
        ).dropna()
        if len(data) < _MIN_OBS:
            continue
        y = data["y"] - data["RF"]
        x = sm.add_constant(data[factor_cols])
        try:
            res = sm.OLS(y, x).fit()
        except Exception as e:
            log.warning("Factor regression failed for %s: %s", name, e)
            continue
        row = {
            "Asset": name,
            "const_coef": round(float(res.params.get("const", 0.0)), 5),
            "const_t": round(float(res.tvalues.get("const", 0.0)), 2),
        }
        for f in factor_cols:
            row[f"{f}_coef"] = round(float(res.params.get(f, 0.0)), 3)
            row[f"{f}_t"] = round(float(res.tvalues.get(f, 0.0)), 2)
        row["R2"] = round(float(res.rsquared), 3)
        row["N"] = int(res.nobs)
        rows.append(row)

    if not rows:
        return pd.DataFrame()
    order = (["Asset", "const_coef", "const_t"]
             + [c for f in factor_cols for c in (f"{f}_coef", f"{f}_t")] + ["R2", "N"])
    return pd.DataFrame(rows)[order]
