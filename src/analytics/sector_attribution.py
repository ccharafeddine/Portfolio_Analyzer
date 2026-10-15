"""Sector-level Brinson-Fachler attribution against an equity benchmark.

Design
------
* Monthly periods. Portfolio sector weights are the start-of-month drifted
  weights from the backtest; portfolio sector returns are the weight-averaged
  member returns within each sector.
* Benchmark sector weights come from the fund's own sector breakdown
  (yfinance ``funds_data``) or, failing that, ``SPY_SECTOR_SNAPSHOT``; they are
  held constant across the window (documented simplification).
* Benchmark sector returns are proxied by the Select Sector SPDR ETFs.
* Holdings outside the benchmark's sectors (bonds, gold, cash ...) form their
  own sector with zero benchmark weight; their benchmark sector return is set
  to their own return, so they contribute pure allocation and no selection.
  Likewise a benchmark sector the portfolio does not hold takes the
  benchmark sector return as its portfolio return (allocation only).
* Monthly effects are linked with Carino (1999) smoothing so the linked
  effects sum exactly to the geometric active return over the window.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

SECTOR_ETFS: dict[str, str] = {
    "Technology": "XLK",
    "Communication Services": "XLC",
    "Consumer Cyclical": "XLY",
    "Consumer Defensive": "XLP",
    "Financial Services": "XLF",
    "Healthcare": "XLV",
    "Industrials": "XLI",
    "Energy": "XLE",
    "Basic Materials": "XLB",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
}

# yfinance funds_data keys -> sector names used by yfinance Ticker.info["sector"]
_FUND_KEYS = {
    "technology": "Technology", "communication_services": "Communication Services",
    "consumer_cyclical": "Consumer Cyclical", "consumer_defensive": "Consumer Defensive",
    "financial_services": "Financial Services", "healthcare": "Healthcare",
    "industrials": "Industrials", "energy": "Energy", "basic_materials": "Basic Materials",
    "utilities": "Utilities", "realestate": "Real Estate",
}

# SPY sector weights (yfinance funds_data, captured 2026-10-15). Fallback only.
SPY_SECTOR_SNAPSHOT: dict[str, float] = {
    "Technology": 0.4045, "Financial Services": 0.1120, "Communication Services": 0.0995,
    "Healthcare": 0.0926, "Consumer Cyclical": 0.0874, "Industrials": 0.0754,
    "Consumer Defensive": 0.0433, "Energy": 0.0340, "Utilities": 0.0188,
    "Real Estate": 0.0170, "Basic Materials": 0.0156,
}

# Common non-equity / ETF holdings that yfinance leaves without a sector.
KNOWN_ASSET_CLASSES: dict[str, str] = {
    "GLD": "Commodities", "IAU": "Commodities", "SLV": "Commodities", "DBC": "Commodities",
    "TLT": "Fixed Income", "IEF": "Fixed Income", "SHY": "Fixed Income", "AGG": "Fixed Income",
    "BND": "Fixed Income", "LQD": "Fixed Income", "HYG": "Fixed Income", "TIP": "Fixed Income",
    "BIL": "Cash", "SGOV": "Cash", "Cash": "Cash",
    "BTC-USD": "Crypto", "ETH-USD": "Crypto",
}


def fund_sector_weights(fund: str) -> tuple[dict[str, float], str]:
    """(weights, source) — live from yfinance when possible, else the snapshot."""
    try:
        import yfinance as yf
        raw = yf.Ticker(fund).funds_data.sector_weightings or {}
        w = {_FUND_KEYS[k]: float(v) for k, v in raw.items() if k in _FUND_KEYS and v}
        tot = sum(w.values())
        if tot > 0.5:
            return {k: v / tot for k, v in w.items()}, f"{fund} holdings (yfinance)"
    except Exception:
        pass
    tot = sum(SPY_SECTOR_SNAPSHOT.values())
    return {k: v / tot for k, v in SPY_SECTOR_SNAPSHOT.items()}, "SPY snapshot (2026-10)"


def classify(ticker: str, sector: str | None) -> str:
    if ticker in KNOWN_ASSET_CLASSES:
        return KNOWN_ASSET_CLASSES[ticker]
    if sector and sector in SECTOR_ETFS:
        return sector
    return "Other"


def carino_factors(port_r: pd.Series, bench_r: pd.Series) -> pd.Series:
    """Per-period Carino smoothing coefficients k_t / k."""
    R = float(np.prod(1 + port_r) - 1)
    B = float(np.prod(1 + bench_r) - 1)

    def _k(a, b):
        if abs(a - b) < 1e-12:
            return 1.0 / (1.0 + a)
        return (math.log1p(a) - math.log1p(b)) / (a - b)

    K = _k(R, B)
    kt = pd.Series([_k(a, b) for a, b in zip(port_r, bench_r)], index=port_r.index)
    return kt / K


def sector_brinson(
    weights_m: pd.DataFrame,
    asset_rets_m: pd.DataFrame,
    sector_of: dict[str, str],
    bench_w: dict[str, float],
    bench_sector_rets_m: pd.DataFrame,
) -> dict:
    """Linked sector attribution.

    weights_m : start-of-period portfolio weights (rows = periods, cols = assets)
    asset_rets_m : period returns per asset (same index)
    sector_of : asset -> sector
    bench_w : benchmark sector weights (sum 1)
    bench_sector_rets_m : period returns per benchmark sector (cols = sector names)

    Returns {"table": per-sector DataFrame, "periods": per-period DataFrame,
             "port_return", "bench_return", "active_return"}.
    """
    idx = weights_m.index.intersection(asset_rets_m.index).intersection(bench_sector_rets_m.index)
    assets = [a for a in weights_m.columns if a in asset_rets_m.columns]
    sectors = sorted(set(bench_w) | {sector_of.get(a, "Other") for a in assets})
    eff_rows, per_rows = [], []
    for t in idx:
        w = weights_m.loc[t, assets].fillna(0.0)
        w = w / w.sum() if w.sum() > 0 else w
        r = asset_rets_m.loc[t, assets].fillna(0.0)
        wp = pd.Series(0.0, index=sectors)
        rp = pd.Series(0.0, index=sectors)
        for s in sectors:
            members = [a for a in assets if sector_of.get(a, "Other") == s]
            ws = float(w[members].sum()) if members else 0.0
            wp[s] = ws
            if ws > 0:
                rp[s] = float((w[members] * r[members]).sum() / ws)
        wb = pd.Series({s: bench_w.get(s, 0.0) for s in sectors})
        rb = pd.Series({s: float(bench_sector_rets_m.loc[t, s]) if s in bench_sector_rets_m.columns
                        and pd.notna(bench_sector_rets_m.loc[t, s]) else np.nan for s in sectors})
        # off-benchmark sectors (and missing proxies): benchmark return := own return
        rb = rb.where(rb.notna() & (wb > 0), rp)
        # unheld sectors: no selection decision was made, so rp := rb
        rp = rp.where(wp > 0, rb)
        Rb = float((wb * rb).sum())
        Rp = float((wp * rp).sum())
        alloc = (wp - wb) * (rb - Rb)
        sel = wb * (rp - rb)
        inter = (wp - wb) * (rp - rb)
        eff_rows.append(pd.DataFrame({"date": t, "Sector": sectors, "wp": wp.values, "wb": wb.values,
                                      "rp": rp.values, "rb": rb.values, "Allocation": alloc.values,
                                      "Selection": sel.values, "Interaction": inter.values}))
        per_rows.append((t, Rp, Rb))
    if not per_rows:
        return {"table": pd.DataFrame(), "periods": pd.DataFrame(),
                "port_return": math.nan, "bench_return": math.nan, "active_return": math.nan}
    per = pd.DataFrame(per_rows, columns=["date", "Portfolio", "Benchmark"]).set_index("date")
    coef = carino_factors(per["Portfolio"], per["Benchmark"])
    eff = pd.concat(eff_rows, ignore_index=True)
    eff["c"] = eff["date"].map(coef)
    for col in ("Allocation", "Selection", "Interaction"):
        eff[col + "_l"] = eff[col] * eff["c"]
    g = eff.groupby("Sector")
    table = pd.DataFrame({
        "Port weight": g["wp"].mean(), "Bench weight": g["wb"].mean(),
        "Allocation": g["Allocation_l"].sum(), "Selection": g["Selection_l"].sum(),
        "Interaction": g["Interaction_l"].sum(),
    })
    table["Total"] = table[["Allocation", "Selection", "Interaction"]].sum(axis=1)
    table = table.sort_values("Total", ascending=False).reset_index()
    R = float(np.prod(1 + per["Portfolio"]) - 1)
    B = float(np.prod(1 + per["Benchmark"]) - 1)
    per["Active"] = per["Portfolio"] - per["Benchmark"]
    return {"table": table, "periods": per, "port_return": R, "bench_return": B, "active_return": R - B}
