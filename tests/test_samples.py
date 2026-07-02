"""Tests for first-run sample portfolios."""

from __future__ import annotations

from src.config.models import PortfolioConfig
from src.ui import samples


def test_sample_specs_build_valid_configs():
    cfgs = samples.sample_configs()
    assert len(cfgs) == len(samples.SAMPLE_SPECS)
    for name, cfg in cfgs:
        assert isinstance(cfg, PortfolioConfig)
        assert cfg.tickers and cfg.weights
        assert abs(sum(cfg.weights.values()) - 1.0) < 1e-9  # weights sum to 1
        assert cfg.benchmark
        assert cfg.start_date < cfg.end_date


def test_seed_writes_and_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.setattr(samples.paths, "portfolios_dir", lambda: tmp_path)

    class FakeSettings:
        def __init__(self):
            self.store = {}

        def value(self, key, default=None, type=None):
            return self.store.get(key, default)

        def setValue(self, key, val):
            self.store[key] = val

    s = FakeSettings()
    n = samples.seed_sample_portfolios(s)
    assert n == len(samples.SAMPLE_SPECS)
    files = sorted(p.name for p in tmp_path.glob("*.json"))
    assert files == sorted(f"{name}.json" for name in samples.SAMPLE_SPECS)

    # Second call is a no-op (flag set).
    assert samples.seed_sample_portfolios(s) == 0

    # Saved samples round-trip back to valid configs.
    cfg = PortfolioConfig.load(str(tmp_path / "Classic 60-40.json"))
    assert cfg.benchmark == "SPY"
    assert set(cfg.weights) == {"SPY", "AGG"}


def test_seed_skips_existing_files(tmp_path, monkeypatch):
    monkeypatch.setattr(samples.paths, "portfolios_dir", lambda: tmp_path)
    (tmp_path / "Classic 60-40.json").write_text("{}")  # pre-existing, must not clobber
    n = samples.seed_sample_portfolios(None)  # no settings -> not flag-guarded
    assert n == len(samples.SAMPLE_SPECS) - 1
    assert (tmp_path / "Classic 60-40.json").read_text() == "{}"
