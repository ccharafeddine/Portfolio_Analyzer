"""Value-at-Risk / Expected Shortfall estimators, VaR backtesting, and the
drawdown table.

Conventions: inputs are simple periodic returns; VaR and ES are returned as
*negative* returns (losses), per period, at confidence ``alpha`` (e.g. 0.95).

Methods
-------
historical      empirical quantile; ES = mean of returns at or below VaR
parametric      normal: VaR = mu + sigma * z_(1-alpha); ES = mu - sigma * phi(z)/(1-alpha)
cornish_fisher  modified VaR: z adjusted for sample skew S and excess kurtosis K
                z_cf = z + (z^2-1)S/6 + (z^3-3z)K/24 - (2z^3-5z)S^2/36;
                ES from the historical tail beyond the CF VaR (ES >= VaR enforced)
monte_carlo     ``n_sims`` draws from a Student-t fitted by moments (df from
                kurtosis, floored at 4.1), scaled to the sample mean/std;
                deterministic for a given ``seed``
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

METHODS = ("historical", "parametric", "cornish_fisher", "monte_carlo")
METHOD_LABELS = {
    "historical": "Historical",
    "parametric": "Parametric (normal)",
    "cornish_fisher": "Cornish-Fisher",
    "monte_carlo": "Monte Carlo (Student-t)",
}


def _clean(r) -> np.ndarray:
    a = np.asarray(pd.Series(r, dtype=float).dropna(), float)
    return a


def var_es(returns, alpha: float = 0.95, method: str = "historical",
           n_sims: int = 100_000, seed: int = 7) -> tuple[float, float]:
    r = _clean(returns)
    if len(r) < 10:
        return math.nan, math.nan
    p = 1.0 - alpha
    mu, sd = float(r.mean()), float(r.std(ddof=1))
    if method == "historical":
        var = float(np.quantile(r, p))
        tail = r[r <= var]
        return var, float(tail.mean())
    z = float(stats.norm.ppf(p))
    if method == "parametric":
        return mu + sd * z, mu - sd * float(stats.norm.pdf(z)) / p
    if method == "cornish_fisher":
        s = float(stats.skew(r, bias=False))
        k = float(stats.kurtosis(r, fisher=True, bias=False))
        zcf = (z + (z * z - 1) * s / 6 + (z ** 3 - 3 * z) * k / 24
               - (2 * z ** 3 - 5 * z) * s * s / 36)
        var = mu + sd * zcf
        tail = r[r <= var]
        es = float(tail.mean()) if len(tail) else var
        return var, min(es, var)
    if method == "monte_carlo":
        k = float(stats.kurtosis(r, fisher=True, bias=False))
        df = max(4.1, 6.0 / k + 4.0) if k > 0 else 30.0
        rng = np.random.default_rng(seed)
        t = rng.standard_t(df, n_sims) * math.sqrt((df - 2) / df)  # unit variance
        sims = mu + sd * t
        var = float(np.quantile(sims, p))
        return var, float(sims[sims <= var].mean())
    raise ValueError(f"unknown VaR method {method!r}")


def var_table(returns, alphas=(0.95, 0.99), methods=METHODS) -> pd.DataFrame:
    rows = []
    for m in methods:
        row = {"Method": METHOD_LABELS[m]}
        for a in alphas:
            v, e = var_es(returns, a, m)
            row[f"VaR {a:.0%}"] = v
            row[f"ES {a:.0%}"] = e
        rows.append(row)
    return pd.DataFrame(rows)


@dataclass
class VarBacktest:
    method: str
    alpha: float
    n: int
    breaches: int
    expected: float
    rate: float
    kupiec_lr: float
    kupiec_p: float
    series: pd.DataFrame  # columns: ret, var, breach

    @property
    def verdict(self) -> str:
        if self.n == 0 or math.isnan(self.kupiec_p):
            return "—"
        return "OK" if self.kupiec_p >= 0.05 else ("Too many breaches" if self.rate > 1 - self.alpha
                                                     else "Too few breaches")


def kupiec_pof(n: int, x: int, alpha: float) -> tuple[float, float]:
    """Kupiec proportion-of-failures LR test (chi-square, 1 df)."""
    if n == 0:
        return math.nan, math.nan
    p = 1.0 - alpha
    phat = x / n

    def _ll(prob: float) -> float:
        a = (n - x) * math.log(1 - prob) if x < n else 0.0
        b = x * math.log(prob) if x > 0 else 0.0
        return a + b

    lr = -2.0 * (_ll(p) - (_ll(phat) if 0 < phat < 1 else 0.0))
    lr = max(lr, 0.0)
    return lr, float(1 - stats.chi2.cdf(lr, 1))


def backtest_var(returns, alpha: float = 0.99, method: str = "historical",
                 window: int = 250, step: int = 1) -> VarBacktest:
    """Rolling one-step-ahead VaR forecasts from the previous ``window`` returns;
    a breach is a realised return below the forecast. Monte Carlo uses fewer
    draws per step for speed."""
    s = pd.Series(returns, dtype=float).dropna()
    if step == 1 and method in ("historical", "parametric", "cornish_fisher"):
        return _backtest_fast(s, alpha, method, window)
    rows = []
    for i in range(window, len(s), step):
        hist = s.iloc[i - window:i]
        kw = {"n_sims": 5_000, "seed": i} if method == "monte_carlo" else {}
        v, _ = var_es(hist, alpha, method, **kw)
        rows.append((s.index[i], s.iloc[i], v))
    df = pd.DataFrame(rows, columns=["date", "ret", "var"]).set_index("date")
    df["breach"] = df["ret"] < df["var"]
    n, x = len(df), int(df["breach"].sum())
    lr, pv = kupiec_pof(n, x, alpha)
    return VarBacktest(method, alpha, n, x, n * (1 - alpha), x / n if n else math.nan, lr, pv, df)


def _backtest_fast(s: pd.Series, alpha: float, method: str, window: int) -> VarBacktest:
    """Vectorised rolling forecasts; identical to the per-window ``var_es``
    (pandas' rolling skew/kurt are the same bias-corrected estimators as
    scipy's ``bias=False``; rolling quantile uses linear interpolation like
    ``np.quantile``)."""
    p = 1.0 - alpha
    roll = s.rolling(window)
    if method == "historical":
        var = roll.quantile(p, interpolation="linear")
    else:
        mu, sd = roll.mean(), roll.std(ddof=1)
        z = float(stats.norm.ppf(p))
        if method == "parametric":
            var = mu + sd * z
        else:
            sk, ku = roll.skew(), roll.kurt()
            zcf = (z + (z * z - 1) * sk / 6 + (z ** 3 - 3 * z) * ku / 24
                   - (2 * z ** 3 - 5 * z) * sk * sk / 36)
            var = mu + sd * zcf
    fc = var.shift(1).iloc[window:]
    df = pd.DataFrame({"ret": s.iloc[window:], "var": fc})
    df.index.name = "date"
    df["breach"] = df["ret"] < df["var"]
    n, x = len(df), int(df["breach"].sum())
    lr, pv = kupiec_pof(n, x, alpha)
    return VarBacktest(method, alpha, n, x, n * (1 - alpha), x / n if n else math.nan, lr, pv, df)


def backtest_table(returns, alpha: float = 0.99, window: int = 250,
                   methods=("historical", "parametric", "cornish_fisher")) -> pd.DataFrame:
    rows = []
    for m in methods:
        bt = backtest_var(returns, alpha, m, window)
        rows.append({"Method": METHOD_LABELS[m], "Days": bt.n, "Breaches": bt.breaches,
                     "Expected": bt.expected, "Breach rate": bt.rate,
                     "Kupiec p": bt.kupiec_p, "Verdict": bt.verdict})
    return pd.DataFrame(rows)


# ── Drawdowns ─────────────────────────────────────────────────────────────────


def drawdown_table(values: pd.Series, top: int = 5) -> pd.DataFrame:
    """Top-``top`` drawdowns by depth: Peak, Trough, Recovery (NaT if not yet
    recovered), Depth, Decline days, Recovery days, Length (trading days peak ->
    recovery, or peak -> last date if open)."""
    v = pd.Series(values, dtype=float).dropna().sort_index()
    cols = ["Peak", "Trough", "Recovery", "Depth", "Decline days", "Recovery days", "Length"]
    if len(v) < 2:
        return pd.DataFrame(columns=cols)
    peak = v.cummax()
    dd = v / peak - 1.0
    under = dd < 0
    episodes = []
    i, n, idx = 0, len(v), v.index
    while i < n:
        if not under.iloc[i]:
            i += 1
            continue
        start = i - 1  # last point at the peak
        j = i
        while j < n and under.iloc[j]:
            j += 1
        seg = dd.iloc[i:j]
        trough_pos = i + int(np.argmin(seg.values))
        rec_pos = j if j < n else None
        episodes.append({
            "Peak": idx[start], "Trough": idx[trough_pos],
            "Recovery": idx[rec_pos] if rec_pos is not None else pd.NaT,
            "Depth": float(seg.min()),
            "Decline days": trough_pos - start,
            "Recovery days": (rec_pos - trough_pos) if rec_pos is not None else np.nan,
            "Length": (rec_pos if rec_pos is not None else n - 1) - start,
        })
        i = j
    out = pd.DataFrame(episodes, columns=cols)
    return out.sort_values("Depth").head(top).reset_index(drop=True)
