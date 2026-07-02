"""Tests for buy/sell trade recommendations."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from src import pipeline as pl
from src.analytics.rebalance import trade_recommendations
from src.config.models import PortfolioConfig


PRICES = pd.Series({"A": 100.0, "B": 50.0})


def test_basic_buy_sell():
    cur = pd.Series({"A": 0.6, "B": 0.4})
    df = trade_recommendations(cur, {"A": 0.5, "B": 0.5}, 100_000, PRICES).set_index("Ticker")
    # A is overweight -> sell $10k (100 sh); B underweight -> buy $10k (200 sh).
    assert df.loc["A", "Action"] == "Sell"
    assert df.loc["A", "TradeValue"] == -10_000
    assert df.loc["A", "Shares"] == -100
    assert df.loc["B", "Action"] == "Buy"
    assert df.loc["B", "TradeValue"] == 10_000
    assert df.loc["B", "Shares"] == 200


def test_no_trade_band():
    cur = pd.Series({"A": 0.6, "B": 0.4})
    df = trade_recommendations(cur, {"A": 0.5, "B": 0.5}, 100_000, PRICES, band=0.15)
    assert (df["Action"] == "Hold").all()
    assert (df["TradeValue"] == 0).all()


def test_target_normalized():
    cur = pd.Series({"A": 0.6, "B": 0.4})
    df = trade_recommendations(cur, {"A": 1, "B": 1}, 100_000, PRICES).set_index("Ticker")
    # {1,1} -> {0.5,0.5}
    assert df.loc["A", "TargetWeight"] == 0.5
    assert df.loc["B", "TargetWeight"] == 0.5


def test_new_and_dropped_tickers():
    cur = pd.Series({"A": 1.0})  # only holds A now
    prices = pd.Series({"A": 100.0, "C": 25.0})
    df = trade_recommendations(cur, {"C": 1.0}, 50_000, prices).set_index("Ticker")
    assert df.loc["A", "Action"] == "Sell" and df.loc["A", "TradeValue"] == -50_000
    assert df.loc["C", "Action"] == "Buy" and df.loc["C", "TradeValue"] == 50_000
    assert df.loc["C", "Shares"] == 2000


def test_sorted_by_trade_size():
    cur = pd.Series({"A": 0.5, "B": 0.5})
    df = trade_recommendations(cur, {"A": 0.9, "B": 0.1}, 100_000, PRICES)
    assert df.iloc[0]["TradeValue"] >= 0  # largest-magnitude trade first
    assert df["TradeValue"].abs().is_monotonic_decreasing


def test_pipeline_populates_trades(monkeypatch, tmp_path):
    idx = pd.bdate_range("2021-01-04", periods=400)
    monkeypatch.setattr(pl, "fetch_prices", lambda tickers, start, end, **kw: pd.DataFrame(
        {c: 100 * np.cumprod(1 + np.random.default_rng(4).normal(0.0004, 0.011, 400))
         for c in sorted(set(tickers))}, index=idx))
    cfg = PortfolioConfig(
        tickers=["AAPL", "MSFT", "NVDA"],
        weights={"AAPL": 0.34, "MSFT": 0.33, "NVDA": 0.33}, benchmark="SPY",
        start_date=date(2021, 1, 4), end_date=date(2022, 8, 1),
    )
    p = pl.AnalysisPipeline(cfg, output_dir=str(tmp_path))
    p._fetch_data(); p._build_active(); p._optimize(); p._recommend_trades()
    assert p.results.trade_recos is not None and not p.results.trade_recos.empty
    assert set(["Ticker", "TradeValue", "Shares", "Action"]).issubset(
        p.results.trade_recos.columns)
    assert p.results.trade_recos_orp is not None  # ORP target available
