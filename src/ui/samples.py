"""First-run sample portfolios.

Seeds a few ready-made portfolios into the user's portfolios directory on first launch
so a new user immediately has something to open, run, and compare. Also exposed via
File -> Open Sample. Uses liquid tickers and a fixed historical window.
"""

from __future__ import annotations

from datetime import date

from src.config.models import PortfolioConfig

from . import paths

_START = date(2019, 1, 1)
_END = date(2025, 12, 31)

# name -> (benchmark, {ticker: weight})
SAMPLE_SPECS: dict[str, tuple[str, dict[str, float]]] = {
    "Classic 60-40": ("SPY", {"SPY": 0.60, "AGG": 0.40}),
    "Tech Growth": ("QQQ", {
        "AAPL": 0.25, "MSFT": 0.25, "NVDA": 0.20, "GOOGL": 0.15, "AMZN": 0.15,
    }),
    "All-Weather (lite)": ("SPY", {
        "VTI": 0.30, "TLT": 0.40, "IEF": 0.15, "GLD": 0.075, "DBC": 0.075,
    }),
    "Dividend Income": ("SPY", {
        "SCHD": 0.30, "JNJ": 0.175, "PG": 0.175, "KO": 0.175, "VZ": 0.175,
    }),
}


def build_sample_config(name: str) -> PortfolioConfig:
    benchmark, weights = SAMPLE_SPECS[name]
    return PortfolioConfig(
        tickers=list(weights.keys()),
        weights=dict(weights),
        benchmark=benchmark,
        start_date=_START,
        end_date=_END,
    )


def sample_configs() -> list[tuple[str, PortfolioConfig]]:
    return [(name, build_sample_config(name)) for name in SAMPLE_SPECS]


def seed_sample_portfolios(settings=None) -> int:
    """Write sample portfolios to the portfolios dir on first run. Returns the count
    written. Idempotent: guarded by a QSettings flag and never overwrites existing files.
    """
    if settings is not None and settings.value("samples_seeded", False, type=bool):
        return 0

    out_dir = paths.portfolios_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for name in SAMPLE_SPECS:
        dest = out_dir / f"{name}.json"
        if dest.exists():
            continue
        try:
            build_sample_config(name).save(str(dest))
            written += 1
        except Exception:
            pass

    if settings is not None:
        settings.setValue("samples_seeded", True)
    return written
