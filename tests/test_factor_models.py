"""Tests for Fama-French factor data parsing and factor regressions (no network)."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from src.analytics.factor_models import FACTOR_SETS, run_factor_model
from src.data import factors


SAMPLE = """This file was created using the 202312 CRSP database.

,Mkt-RF,SMB,HML,RF
20220103, 1.00,-0.50, 0.20, 0.01
20220104,-0.30, 0.10, 0.05, 0.01
20220105, 0.40, 0.00,-0.10, 0.01

  Copyright 2023 Kenneth R. French
"""


def test_parse_ff_csv():
    df = factors._parse_ff_csv(SAMPLE)
    assert list(df.columns) == ["Mkt-RF", "SMB", "HML", "RF"]
    assert len(df) == 3
    # Source is percent -> decimals.
    assert df.loc["2022-01-03", "Mkt-RF"] == 0.01
    assert df.loc["2022-01-04", "SMB"] == 0.001


def test_regression_recovers_betas():
    idx = pd.bdate_range("2021-01-04", periods=400)
    rng = np.random.default_rng(0)
    ff = pd.DataFrame({
        "Mkt-RF": rng.normal(0.0003, 0.01, len(idx)),
        "SMB": rng.normal(0.0, 0.006, len(idx)),
        "HML": rng.normal(0.0, 0.006, len(idx)),
        "RF": np.full(len(idx), 0.0001),
    }, index=idx)
    true = {"Mkt-RF": 1.10, "SMB": 0.30, "HML": -0.20}
    alpha = 0.0002
    excess = alpha + sum(true[f] * ff[f] for f in true) + rng.normal(0, 1e-4, len(idx))
    daily = pd.DataFrame({"AAA": excess + ff["RF"]}, index=idx)

    out = run_factor_model(daily, ff, FACTOR_SETS["FF3"])
    row = out.set_index("Asset").loc["AAA"]
    assert abs(row["Mkt-RF_coef"] - 1.10) < 0.05
    assert abs(row["SMB_coef"] - 0.30) < 0.05
    assert abs(row["HML_coef"] + 0.20) < 0.05
    assert row["R2"] > 0.9


def test_portfolio_row_leads():
    idx = pd.bdate_range("2021-01-04", periods=200)
    rng = np.random.default_rng(1)
    ff = pd.DataFrame({
        "Mkt-RF": rng.normal(0, 0.01, len(idx)), "SMB": rng.normal(0, 0.006, len(idx)),
        "HML": rng.normal(0, 0.006, len(idx)), "RF": np.full(len(idx), 0.0001),
    }, index=idx)
    daily = pd.DataFrame({"AAA": rng.normal(0.0005, 0.012, len(idx))}, index=idx)
    port = pd.Series(rng.normal(0.0004, 0.01, len(idx)), index=idx)
    out = run_factor_model(daily, ff, FACTOR_SETS["FF3"], port_returns=port)
    assert out.iloc[0]["Asset"] == "Portfolio"
    assert set(out["Asset"]) == {"Portfolio", "AAA"}


def test_fetch_assembles_carhart(monkeypatch):
    idx = pd.bdate_range("2021-01-04", periods=100)
    ff3 = pd.DataFrame({
        "Mkt-RF": 0.001, "SMB": 0.0, "HML": 0.0, "RF": 0.0001
    }, index=idx)
    mom = pd.DataFrame({"MOM": 0.0002}, index=idx)
    ff5 = pd.DataFrame({
        "Mkt-RF": 0.001, "SMB": 0.0, "HML": 0.0, "RMW": 0.0, "CMA": 0.0, "RF": 0.0001
    }, index=idx)

    def fake_download(key, cache_dir):
        return {"ff3": ff3, "mom": mom, "ff5": ff5}[key]

    monkeypatch.setattr(factors, "_download", fake_download)

    car = factors.fetch_ff_factors(date(2021, 1, 4), date(2021, 6, 1), "Carhart 4-Factor")
    assert "MOM" in car.columns and "Mkt-RF" in car.columns and "RF" in car.columns
    assert car.index.max() <= pd.Timestamp("2021-06-01")

    f5 = factors.fetch_ff_factors(date(2021, 1, 4), date(2021, 6, 1), "FF5")
    assert {"RMW", "CMA"}.issubset(f5.columns)

    assert factors.fetch_ff_factors(date(2021, 1, 4), date(2021, 6, 1), "bogus") is None
