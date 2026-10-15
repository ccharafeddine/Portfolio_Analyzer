from datetime import date

import pandas as pd
import pytest

from src.data.risk_free import discount_to_bey, resolve_risk_free_rate

S, E = date(2024, 1, 1), date(2024, 12, 31)


def _irx(values):
    idx = pd.bdate_range("2024-01-02", periods=len(values))
    return pd.DataFrame({"^IRX": values}, index=idx)


def test_tbill_average_converted_to_bond_equivalent():
    rf = resolve_risk_free_rate(S, E, fallback=0.04, fetch=lambda *a, **k: _irx([5.0, 5.0, 5.0]))
    expected = 365 * 0.05 / (360 - 91 * 0.05)
    assert rf.rate == pytest.approx(expected)
    assert rf.source.startswith("^IRX") and rf.warning is None


def test_manual_source_skips_fetch():
    def boom(*a, **k):
        raise AssertionError("should not fetch")
    rf = resolve_risk_free_rate(S, E, fallback=0.031, source="manual", fetch=boom)
    assert rf.rate == 0.031 and rf.source == "manual"


@pytest.mark.parametrize("fetch", [
    lambda *a, **k: (_ for _ in ()).throw(ConnectionError("offline")),
    lambda *a, **k: pd.DataFrame(),
    lambda *a, **k: _irx([float("nan")] * 3),
])
def test_falls_back_with_warning(fetch):
    rf = resolve_risk_free_rate(S, E, fallback=0.04, fetch=fetch)
    assert rf.rate == 0.04
    assert rf.source.startswith("fallback")
    assert "^IRX" in rf.warning


def test_discount_to_bey_is_above_discount_yield():
    assert (discount_to_bey(pd.Series([2.0, 5.0])) > pd.Series([0.02, 0.05])).all()
