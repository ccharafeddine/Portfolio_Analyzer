import numpy as np
import pandas as pd

from src.data.quality import quality_report


def _frame():
    idx = pd.bdate_range("2024-01-01", periods=60)
    g = np.random.default_rng(0)
    base = 100 * np.cumprod(1 + g.normal(0, 0.01, 60))
    df = pd.DataFrame({"OK": base, "GAP": base.copy(), "SPLIT": base.copy(),
                       "LATE": base.copy(), "STALE": base.copy(), "FLAT": base.copy()}, idx)
    df.iloc[20:25, 1] = np.nan            # 5-day gap
    df.iloc[30:, 2] = df.iloc[30:, 2] / 4  # unadjusted 4:1 split
    df.iloc[:15, 3] = np.nan              # late inception (not a gap)
    df.iloc[50:, 4] = np.nan              # stale for 10 days
    df.iloc[10:25, 5] = df.iloc[10, 5]    # flat feed
    return df


def test_quality_flags():
    rep = quality_report(_frame(), start="2024-01-01")
    st = rep.table.set_index("Ticker")["Status"]
    assert st["OK"] == "OK"
    assert st["GAP"] == "Gaps" and rep.table.set_index("Ticker").loc["GAP", "Longest gap"] == 5
    assert "Split?" in st["SPLIT"]
    assert st["LATE"] == "Inception"
    assert rep.table.set_index("Ticker").loc["LATE", "Gaps"] == 0
    assert st["STALE"] == "Stale"
    assert "Flat" in st["FLAT"]
    assert rep.gaps == 1 and len(rep.issues) == 5


def test_empty():
    rep = quality_report(pd.DataFrame())
    assert rep.ok and rep.days == 0
