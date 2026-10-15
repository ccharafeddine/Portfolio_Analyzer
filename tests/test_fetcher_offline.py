from datetime import date

import pandas as pd
import pytest

from src.data import fetcher


def _boom(*a, **k):
    raise OSError("network unreachable")


def test_offline_uses_stale_cache(tmp_path, monkeypatch):
    idx = pd.bdate_range("2024-01-01", "2024-03-29")
    df = pd.DataFrame({"AAA": range(len(idx)), "BBB": range(len(idx))}, idx, dtype=float)
    df.to_parquet(tmp_path / "prices_other.parquet")  # a superset request cached earlier
    monkeypatch.setattr(fetcher.yf, "download", _boom)
    out = fetcher.fetch_prices(["AAA"], date(2024, 1, 1), date(2024, 2, 29), cache_dir=tmp_path)
    assert list(out.columns) == ["AAA"] and out.index[-1] <= pd.Timestamp("2024-02-29")
    assert out.attrs["offline"] is True


def test_offline_without_cache_raises_clear_error(tmp_path, monkeypatch):
    monkeypatch.setattr(fetcher.yf, "download", _boom)
    with pytest.raises(ValueError, match="internet connection"):
        fetcher.fetch_prices(["AAA"], date(2024, 1, 1), date(2024, 2, 1), cache_dir=tmp_path)
