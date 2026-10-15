"""Pin src.analytics.metrics against reference implementations.

Reference values were produced with empyrical-reloaded 0.5.12 (and statsmodels
OLS for standard errors) on ``fixtures/metrics_returns.csv`` with rf = 4% p.a.
de-annualized geometrically to a daily rate. If empyrical is installed the
values are also re-derived live.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.analytics import metrics as M
from src.analytics.risk import tail_metrics

FIX = pd.read_csv(Path(__file__).parent / "fixtures" / "metrics_returns.csv")
R, B = FIX["port"], FIX["bench"]
RF = 0.04
RFP = (1 + RF) ** (1 / 252) - 1

# empyrical-reloaded 0.5.12 / statsmodels 0.14 on the fixture above
REF = {
    "ann_ret": -0.028225747321868888,
    "vol": 0.18757388857057877,
    "sharpe": -0.2680823693158461,
    "sortino": -0.374492530169199,
    "downside": 0.13427571558521595,
    "mdd": -0.27854661389997454,
    "calmar": -0.1013322220172624,
    "beta": 0.71871696155822,
    "alpha_ann_compounded": -0.05787913455806637,
    "alpha_se": 0.0003286996512220775,
    "beta_se": 0.03085574918522528,
    "t_alpha": -0.7197028712759891,
    "t_beta": 23.292805410226897,
    "r2": 0.42040430259618655,
}
TOL = dict(rel=1e-6, abs=1e-6)


def test_return_and_vol():
    assert M.annualized_return(R) == pytest.approx(REF["ann_ret"], **TOL)
    assert M.annualized_volatility(R) == pytest.approx(REF["vol"], **TOL)


def test_sharpe_sortino_share_rf_hurdle():
    assert M.sharpe_ratio(R, RF) == pytest.approx(REF["sharpe"], **TOL)
    assert M.sortino_ratio(R, RF) == pytest.approx(REF["sortino"], **TOL)
    assert M.downside_deviation(R, RF) == pytest.approx(REF["downside"], **TOL)


def test_drawdown_and_calmar():
    assert M.max_drawdown_from_returns(R) == pytest.approx(REF["mdd"], **TOL)
    assert M.calmar_ratio(R) == pytest.approx(REF["calmar"], **TOL)


def test_capm_with_standard_errors():
    c = M.capm(R, B, RF, M.DAILY)
    assert c.beta == pytest.approx(REF["beta"], **TOL)
    assert (1 + c.alpha) ** 252 - 1 == pytest.approx(REF["alpha_ann_compounded"], **TOL)
    assert c.alpha_annual == pytest.approx(c.alpha * 252)
    assert c.alpha_se == pytest.approx(REF["alpha_se"], **TOL)
    assert c.beta_se == pytest.approx(REF["beta_se"], **TOL)
    assert c.t_alpha == pytest.approx(REF["t_alpha"], **TOL)
    assert c.t_beta == pytest.approx(REF["t_beta"], **TOL)
    assert c.r_squared == pytest.approx(REF["r2"], **TOL)


def test_sortino_uses_all_periods_not_just_losers():
    r = pd.Series([0.02, -0.01, 0.03, -0.02, 0.01])
    dd = np.sqrt(np.mean(np.minimum(r, 0) ** 2)) * np.sqrt(252)
    assert M.downside_deviation(r) == pytest.approx(dd)
    # the old (wrong) definition: std of the negative days only
    wrong = r[r < 0].std() * np.sqrt(252)
    assert M.downside_deviation(r) != pytest.approx(wrong)


def test_tail_metrics_sortino_matches_metrics_module():
    t = tail_metrics(R, rf_annual=RF)
    assert t["Sortino"] == pytest.approx(REF["sortino"], **TOL)
    assert t["Calmar"] == pytest.approx(REF["calmar"], **TOL)


def test_periodicity_helpers():
    idx_d = pd.bdate_range("2024-01-01", periods=30)
    idx_m = pd.date_range("2020-01-31", periods=30, freq="ME")
    assert M.infer_periods_per_year(idx_d) == M.DAILY
    assert M.infer_periods_per_year(idx_m) == M.MONTHLY
    assert (1 + M.periodic_rate(0.05, 12)) ** 12 == pytest.approx(1.05)


def test_live_against_empyrical_if_installed():
    ep = pytest.importorskip("empyrical")
    assert M.sortino_ratio(R, RF) == pytest.approx(
        ep.sortino_ratio(R, required_return=RFP, annualization=252), **TOL)
    assert M.sharpe_ratio(R, RF) == pytest.approx(
        ep.sharpe_ratio(R, risk_free=RFP, annualization=252), **TOL)
    assert M.calmar_ratio(R) == pytest.approx(ep.calmar_ratio(R, annualization=252), **TOL)


def test_rolling_sharpe_matches_point_sharpe_on_last_window():
    roll = M.rolling_sharpe(R, window=252, rf_annual=RF)
    assert roll.iloc[-1] == pytest.approx(M.sharpe_ratio(R.iloc[-252:], RF), **TOL)
    assert roll.iloc[:251].isna().all()


def test_rolling_vol_and_beta():
    assert M.rolling_volatility(R, 63).iloc[-1] == pytest.approx(
        M.annualized_volatility(R.iloc[-63:]), **TOL)
    c = M.capm(R.iloc[-63:], B.iloc[-63:], 0.0, M.DAILY)
    assert M.rolling_beta(R, B, 63).iloc[-1] == pytest.approx(c.beta, **TOL)


def test_ex_ante_sharpe_and_turnover():
    w, mu = np.array([0.6, 0.4]), np.array([0.10, 0.04])
    cov = np.array([[0.04, 0.006], [0.006, 0.01]])
    vol = np.sqrt(w @ cov @ w)
    assert M.ex_ante_sharpe(w, mu, cov, 0.02) == pytest.approx((0.076 - 0.02) / vol)
    from src.analytics.optimization import sharpe_ratio as opt_sharpe
    assert opt_sharpe(w, mu, cov / 12.0, 0.02) == pytest.approx(M.ex_ante_sharpe(w, mu, cov, 0.02))
    assert M.turnover({"A": 0.5, "B": 0.5}, {"A": 0.2, "C": 0.8}) == pytest.approx(0.8)
