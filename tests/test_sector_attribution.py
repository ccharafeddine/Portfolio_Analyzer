import numpy as np
import pandas as pd
import pytest

from src.analytics import sector_attribution as S
from src.analytics.attribution import brinson_fachler


def _setup(n=6, seed=1):
    g = np.random.default_rng(seed)
    idx = pd.period_range("2024-01", periods=n, freq="M").to_timestamp("M")
    assets = ["AAA", "BBB", "CCC", "GLD"]
    rets = pd.DataFrame(g.normal(0.01, 0.05, (n, 4)), idx, assets)
    w = pd.DataFrame(g.dirichlet(np.ones(4), n), idx, assets)
    sector_of = {"AAA": "Technology", "BBB": "Technology", "CCC": "Healthcare", "GLD": "Commodities"}
    bw = {"Technology": 0.6, "Healthcare": 0.25, "Energy": 0.15}
    br = pd.DataFrame(g.normal(0.008, 0.04, (n, 3)), idx, list(bw))
    return w, rets, sector_of, bw, br


def test_single_period_matches_brinson_fachler():
    w, r, so, bw, br = _setup(n=1)
    out = S.sector_brinson(w, r, so, bw, br)
    t = out["table"].set_index("Sector")
    # rebuild sector-level inputs and compare with the generic BF routine
    wp = w.iloc[0].groupby(so).sum()
    rp = (w.iloc[0] * r.iloc[0]).groupby(so).sum() / wp
    rb = br.iloc[0].reindex(wp.index.union(br.columns))
    rb["Commodities"] = rp["Commodities"]
    rp = rp.reindex(rb.index).fillna(rb)  # unheld benchmark sectors take rb
    ref = brinson_fachler(wp, pd.Series(bw), rp, rb).set_index("Asset")
    for c in ("Allocation", "Selection", "Interaction"):
        np.testing.assert_allclose(t[c].reindex(ref.index).values, ref[c].values, atol=1e-12)
    assert t.loc["Commodities", "Selection"] == pytest.approx(0.0)


def test_carino_linked_effects_sum_to_geometric_active_return():
    out = S.sector_brinson(*_setup(n=24))
    tot = out["table"][["Allocation", "Selection", "Interaction"]].to_numpy().sum()
    assert tot == pytest.approx(out["active_return"], abs=1e-12)
    assert out["active_return"] == pytest.approx(out["port_return"] - out["bench_return"])


def test_classify_and_snapshot():
    assert S.classify("TLT", "") == "Fixed Income"
    assert S.classify("AAPL", "Technology") == "Technology"
    assert S.classify("XYZ", "Unknown") == "Other"
    assert sum(S.SPY_SECTOR_SNAPSHOT.values()) == pytest.approx(1.0, abs=0.01)


def test_unheld_sector_is_pure_allocation():
    w, r, so, bw, br = _setup(n=12)
    out = S.sector_brinson(w, r, so, bw, br)["table"].set_index("Sector")
    assert out.loc["Energy", "Port weight"] == 0
    assert out.loc["Energy", "Selection"] == pytest.approx(0.0)
    assert out.loc["Energy", "Interaction"] == pytest.approx(0.0)
    assert out.loc["Energy", "Allocation"] != 0
