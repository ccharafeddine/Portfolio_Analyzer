"""Shared fixtures. ``offline_results`` runs the full pipeline on synthetic
prices with every network source stubbed, so view and report tests can render
real AnalysisResults in CI without internet access."""

from __future__ import annotations

import zlib
from datetime import date

import numpy as np
import pandas as pd
import pytest


def _synthetic_prices(tickers, start, end, **_):
    idx = pd.bdate_range(pd.Timestamp(start), pd.Timestamp(end))
    cols = {}
    for t in tickers:
        g = np.random.default_rng(zlib.crc32(t.encode()))
        drift = 0.0006 if t not in ("TLT", "^IRX") else 0.0001
        cols[t] = 100 * np.cumprod(1 + g.normal(drift, 0.012, len(idx)))
    df = pd.DataFrame(cols, idx)
    df.index.name = "Date"
    return df


def _synthetic_ff(start, end, model="FF3", **_):
    idx = pd.bdate_range(pd.Timestamp(start), pd.Timestamp(end))
    g = np.random.default_rng(1)
    cols = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "MOM"]
    df = pd.DataFrame(g.normal(0, 0.008, (len(idx), len(cols))), idx, cols)
    df["RF"] = 0.0001
    return df


@pytest.fixture(scope="session")
def offline_results(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    import src.analytics.income as income
    import src.analytics.sector_attribution as SA
    import src.data.factors as factors
    import src.data.meta_cache as meta
    import src.pipeline as pipeline
    from src.config.models import PortfolioConfig

    mp.setattr(pipeline, "fetch_prices", _synthetic_prices)
    mp.setattr(income, "_dividend_history", lambda t: pd.Series(dtype=float))
    mp.setattr(meta, "get_sectors", lambda tickers, **k: {t: "Technology" for t in tickers})
    mp.setattr(SA, "fund_sector_weights", lambda fund: (dict(SA.SPY_SECTOR_SNAPSHOT), "snapshot"))
    mp.setattr(factors, "fetch_ff_factors", _synthetic_ff)
    mp.setattr("src.analytics.exposure.yf", None, raising=False)
    try:
        cfg = PortfolioConfig(
            tickers=["AAA", "BBB", "GLD", "TLT"],
            weights={"AAA": 0.3, "BBB": 0.3, "GLD": 0.2, "TLT": 0.2},
            benchmark="SPY", start_date=date(2021, 1, 1), end_date=date(2024, 12, 31),
            risk_free_source="manual", risk_free_rate=0.03,
        )
        out = tmp_path_factory.mktemp("pa_out")
        res = pipeline.AnalysisPipeline(cfg, output_dir=str(out)).run()
    finally:
        mp.undo()
    return res
