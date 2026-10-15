import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src.analytics import tail_risk as T

FIX = pd.read_csv("tests/fixtures/metrics_returns.csv")["port"]


def test_parametric_matches_quantstats_reference():
    # quantstats.stats.value_at_risk / conditional_value_at_risk(sigma=1, confidence=.95)
    v, e = T.var_es(FIX, 0.95, "parametric")
    assert v == pytest.approx(-0.01947955815402089, abs=1e-12)
    assert e == pytest.approx(-0.02441700075279673, abs=1e-12)


def test_historical_is_empirical_quantile():
    v, e = T.var_es(FIX, 0.95, "historical")
    assert v == pytest.approx(np.quantile(FIX, 0.05))
    assert e == pytest.approx(FIX[FIX <= v].mean())


def test_cornish_fisher_formula_and_normal_limit():
    r = FIX.values
    z = stats.norm.ppf(0.01)
    s, k = stats.skew(r, bias=False), stats.kurtosis(r, bias=False)
    zcf = z + (z*z-1)*s/6 + (z**3-3*z)*k/24 - (2*z**3-5*z)*s*s/36
    v, e = T.var_es(r, 0.99, "cornish_fisher")
    assert v == pytest.approx(r.mean() + r.std(ddof=1) * zcf)
    assert e <= v
    g = np.random.default_rng(0).normal(0, 0.01, 200_000)
    assert T.var_es(g, 0.99, "cornish_fisher")[0] == pytest.approx(T.var_es(g, 0.99, "parametric")[0], rel=0.01)


def test_monte_carlo_deterministic_and_reasonable():
    a = T.var_es(FIX, 0.99, "monte_carlo")
    assert a == T.var_es(FIX, 0.99, "monte_carlo")
    h = T.var_es(FIX, 0.99, "historical")[0]
    assert a[0] == pytest.approx(h, rel=0.35)
    assert a[1] < a[0]


def test_kupiec_known_value():
    lr, p = T.kupiec_pof(250, 8, 0.99)
    # closed form: -2 ln[(0.99^242 0.01^8) / ((242/250)^242 (8/250)^8)]
    ref = -2 * (242*math.log(.99) + 8*math.log(.01) - 242*math.log(242/250) - 8*math.log(8/250))
    assert lr == pytest.approx(ref)
    assert p == pytest.approx(1 - stats.chi2.cdf(ref, 1))
    assert T.kupiec_pof(250, 0, 0.99)[0] == pytest.approx(-2 * 250 * math.log(.99))


def test_backtest_counts_breaches_against_prior_window_only():
    s = pd.Series(np.r_[np.full(300, 0.001), -0.5, np.full(49, 0.001)])
    bt = T.backtest_var(s, 0.99, "historical", window=250)
    assert bt.n == 100 and bt.breaches == 1
    assert bool(bt.series["breach"].iloc[50])


def test_drawdown_table():
    idx = pd.bdate_range("2024-01-01", periods=9)
    v = pd.Series([100, 90, 80, 100, 110, 99, 105, 111, 100], idx)
    t = T.drawdown_table(v, top=5)
    assert len(t) == 3
    first = t.iloc[0]
    assert first["Depth"] == pytest.approx(-0.2)
    assert (first["Peak"], first["Trough"], first["Recovery"]) == (idx[0], idx[2], idx[3])
    assert (first["Decline days"], first["Recovery days"], first["Length"]) == (2, 1, 3)
    open_dd = t[t["Recovery"].isna()].iloc[0]
    assert open_dd["Peak"] == idx[7] and open_dd["Length"] == 1


def test_stress_replay_and_beta_shocks():
    from src.analytics import risk as R
    idx = pd.bdate_range("2008-01-01", periods=40)
    px = pd.DataFrame({"A": np.linspace(100, 50, 40), "B": np.full(40, 10.0)}, idx)
    vals, cover = R.replay_values(px, {"A": 0.5, "B": 0.5, "Z": 0.0}, idx[0], idx[-1])
    assert cover == 1.0 and vals.iloc[-1] == pytest.approx(0.75)
    hist = pd.Series(1.0, pd.bdate_range("2020-01-01", periods=5))
    res = R.run_stress_tests(hist, hist, [("X", str(idx[0].date()), str(idx[-1].date()), "")],
                             asset_prices=px, weights={"A": 0.5, "B": 0.5},
                             benchmark_prices=px["A"])
    assert res[0].basis == "Replay" and res[0].portfolio_return == pytest.approx(-0.25)
    assert res[0].benchmark_return == pytest.approx(-0.5)
    g = np.random.default_rng(3)
    b = pd.Series(g.normal(0, 0.01, 500))
    p = 1.5 * b + g.normal(0, 0.001, 500)
    t = R.beta_shock_table(p, b, 1000.0, shocks=(-0.1,))
    assert t["Beta"].iloc[0] == pytest.approx(1.5, abs=0.02)
    assert t["P&L"].iloc[0] == pytest.approx(t["Portfolio"].iloc[0] * 1000)
