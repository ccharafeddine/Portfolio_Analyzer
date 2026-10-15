"""Pure table builders behind the redesigned Overview / Performance pages."""

import numpy as np
import pandas as pd
import pytest

from src.analytics.summary import data_quality, holdings_summary, monthly_calendar, period_returns


def _prices():
    idx = pd.bdate_range("2021-01-01", periods=800)
    rng = np.random.default_rng(1)
    data = {t: 100 * np.exp(np.cumsum(rng.normal(m, 0.012, len(idx))))
            for t, m in (("AAA", 0.0006), ("BBB", 0.0002), ("SPY", 0.0004))}
    return pd.DataFrame(data, index=idx)


def _holdings(prices):
    p0 = prices.iloc[0]
    return pd.DataFrame({
        "Ticker": ["AAA", "BBB"], "TargetWeight": [0.6, 0.4],
        "PurchasePrice": [p0["AAA"], p0["BBB"]],
        "Shares": [600_000 / p0["AAA"], 400_000 / p0["BBB"]],
        "Invested": [600_000.0, 400_000.0], "RealizedWeight": [0.6, 0.4],
    })


def test_holdings_summary_totals_are_consistent():
    px = _prices()
    hs = holdings_summary(_holdings(px), px, "SPY")
    body, total = hs.iloc[:-1], hs.iloc[-1]
    assert total["Ticker"] == "Portfolio"
    assert body["Weight"].sum() == pytest.approx(1.0)
    assert total["Contribution"] == pytest.approx(total["Return"])  # contributions add up
    assert body["RiskContribution"].sum() == pytest.approx(1.0)
    assert body["Weight"].is_monotonic_decreasing
    assert not body["Beta"].isna().any()


def test_holdings_summary_handles_missing_inputs():
    assert holdings_summary(None, _prices()).empty
    assert holdings_summary(_holdings(_prices()), pd.DataFrame()).empty


def test_period_returns_rows_and_excess():
    px = _prices()
    pr = period_returns(px["AAA"], px["SPY"])
    assert list(pr["Period"]) == ["1M", "3M", "YTD", "1Y", "3Y ann.", "Since incep. ann."]
    assert (pr["Excess"] - (pr["Portfolio"] - pr["Benchmark"])).abs().max() < 1e-12
    one_year = px["AAA"].iloc[-1] / px["AAA"][px.index < px.index[-1] - pd.DateOffset(years=1)].iloc[-1] - 1
    assert pr.loc[pr["Period"] == "1Y", "Portfolio"].iloc[0] == pytest.approx(one_year)


def test_monthly_calendar_compounds_to_year_total():
    px = _prices()
    cal = monthly_calendar(px["AAA"], px["SPY"])
    months = cal.columns[:12]
    for year, row in cal.iterrows():
        assert (1 + row[months].dropna()).prod() - 1 == pytest.approx(row["Year total"])
    assert "Benchmark" in cal.columns


def test_data_quality_counts_gaps():
    px = _prices()
    px.iloc[5, 0] = np.nan
    q = data_quality(px)
    assert q["gaps"] == 1 and q["days"] == len(px) and q["as_of"] == px.index[-1]
