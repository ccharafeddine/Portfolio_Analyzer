"""Company fundamentals for the Fundamentals tab.

yfinance baseline (no key) via ``Ticker.info`` + ``Ticker.calendar``, optionally
enriched with FMP's discounted-cash-flow fair value when an FMP key is set. Qt-free,
disk-cached (~6h), and defensive: a per-ticker failure yields a sparse record rather
than raising.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Optional

try:
    import requests
except Exception:  # pragma: no cover
    requests = None

try:
    import yfinance as yf
except Exception:  # pragma: no cover
    yf = None


_CACHE_DIR = Path(
    os.getenv("PORTFOLIO_ANALYZER_CACHE_DIR")
    or (Path.home() / ".portfolio_analyzer_cache")
).parent / "market"
CACHE_TTL = 6 * 60 * 60  # 6 hours


def _cache_file(name: str) -> Path:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return _CACHE_DIR / name


def _load_cache(name: str, ttl: int):
    try:
        p = _cache_file(name)
        if p.exists() and (time.time() - p.stat().st_mtime) < ttl:
            return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        pass
    return None


def _save_cache(name: str, obj) -> None:
    try:
        _cache_file(name).write_text(json.dumps(obj, default=str), encoding="utf-8")
    except Exception:
        pass


def _f(v) -> Optional[float]:
    try:
        x = float(v)
        return x if x == x else None
    except (TypeError, ValueError):
        return None


# Fundamentals field -> yfinance ``info`` key
_INFO_MAP = {
    "name": "longName",
    "sector": "sector",
    "industry": "industry",
    "market_cap": "marketCap",
    "price": "currentPrice",
    "pe": "trailingPE",
    "forward_pe": "forwardPE",
    "pb": "priceToBook",
    "ps": "priceToSalesTrailing12Months",
    "peg": "pegRatio",
    "ev_ebitda": "enterpriseToEbitda",
    "gross_margin": "grossMargins",
    "operating_margin": "operatingMargins",
    "net_margin": "profitMargins",
    "roe": "returnOnEquity",
    "roa": "returnOnAssets",
    "revenue_growth": "revenueGrowth",
    "earnings_growth": "earningsGrowth",
    "debt_to_equity": "debtToEquity",
    "current_ratio": "currentRatio",
    "dividend_yield": "dividendYield",   # yfinance returns this already in percent
    "payout_ratio": "payoutRatio",
    "beta": "beta",
    "eps_ttm": "trailingEps",
    "eps_forward": "forwardEps",
}


@dataclass
class Fundamentals:
    ticker: str
    name: str = ""
    sector: str = ""
    industry: str = ""
    market_cap: Optional[float] = None
    price: Optional[float] = None
    pe: Optional[float] = None
    forward_pe: Optional[float] = None
    pb: Optional[float] = None
    ps: Optional[float] = None
    peg: Optional[float] = None
    ev_ebitda: Optional[float] = None
    gross_margin: Optional[float] = None
    operating_margin: Optional[float] = None
    net_margin: Optional[float] = None
    roe: Optional[float] = None
    roa: Optional[float] = None
    revenue_growth: Optional[float] = None
    earnings_growth: Optional[float] = None
    debt_to_equity: Optional[float] = None
    current_ratio: Optional[float] = None
    dividend_yield: Optional[float] = None
    payout_ratio: Optional[float] = None
    beta: Optional[float] = None
    eps_ttm: Optional[float] = None
    eps_forward: Optional[float] = None
    next_earnings: Optional[str] = None   # ISO date
    ex_dividend: Optional[str] = None     # ISO date
    dcf: Optional[float] = None           # FMP discounted-cash-flow fair value

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Fundamentals":
        return cls(**d)


def _iso(value) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        value = value[0] if value else None
    if isinstance(value, (date, datetime)):
        return value.isoformat()[:10]
    s = str(value).strip()
    return s[:10] if s else None


def _from_yfinance(ticker: str) -> Fundamentals:
    f = Fundamentals(ticker=ticker, name=ticker)
    if yf is None:
        return f
    try:
        tk = yf.Ticker(ticker)
        info = tk.info or {}
        for attr, key in _INFO_MAP.items():
            val = info.get(key)
            if val is None:
                continue
            if attr in ("name", "sector", "industry"):
                setattr(f, attr, str(val))
            else:
                setattr(f, attr, _f(val))
        if not f.name:
            f.name = info.get("shortName") or ticker
    except Exception:
        pass
    try:
        cal = yf.Ticker(ticker).calendar or {}
        f.next_earnings = _iso(cal.get("Earnings Date"))
        f.ex_dividend = _iso(cal.get("Ex-Dividend Date"))
    except Exception:
        pass
    return f


def _enrich_fmp(items: list[Fundamentals], key: str) -> None:
    if not (requests and key and items):
        return
    syms = ",".join(i.ticker for i in items)
    resp = requests.get(
        f"https://financialmodelingprep.com/api/v3/profile/{syms}",
        params={"apikey": key},
        timeout=15,
    )
    data = resp.json()
    if not isinstance(data, list):
        return
    by_sym = {d.get("symbol"): d for d in data if isinstance(d, dict)}
    for it in items:
        d = by_sym.get(it.ticker)
        if not d:
            continue
        it.dcf = _f(d.get("dcf")) or it.dcf
        if it.price is None:
            it.price = _f(d.get("price"))
        if not it.sector:
            it.sector = d.get("sector") or ""
        if not it.industry:
            it.industry = d.get("industry") or ""


def fetch_fundamentals(
    tickers, fmp_key: Optional[str] = None, use_cache: bool = True
) -> list[Fundamentals]:
    """Per-holding fundamentals from yfinance, optionally enriched with FMP DCF."""
    tickers = [str(t).upper() for t in tickers if t]
    if not tickers:
        return []
    cache_name = f"fundamentals_{'-'.join(sorted(tickers))}_{'fmp' if fmp_key else 'yf'}.json"
    if use_cache:
        cached = _load_cache(cache_name, CACHE_TTL)
        if cached is not None:
            return [Fundamentals.from_dict(d) for d in cached]

    items = [_from_yfinance(t) for t in tickers]
    if fmp_key:
        try:
            _enrich_fmp(items, fmp_key)
        except Exception:
            pass

    if use_cache and any(i.name and i.name != i.ticker for i in items):
        _save_cache(cache_name, [i.to_dict() for i in items])
    return items


# ── Deeper statements + analyst estimates (per ticker, lazy) ──

_INCOME_ROWS = [
    "Total Revenue", "Gross Profit", "Operating Income", "EBITDA",
    "Net Income", "Diluted EPS",
]
_BALANCE_ROWS = [
    "Total Assets", "Total Liabilities Net Minority Interest",
    "Stockholders Equity", "Total Debt", "Cash And Cash Equivalents",
]
_CASHFLOW_ROWS = [
    "Operating Cash Flow", "Investing Cash Flow", "Financing Cash Flow",
    "Free Cash Flow",
]


def _statement_dict(df, rows, max_periods: int = 5) -> Optional[dict]:
    if df is None or getattr(df, "empty", True):
        return None
    cols = list(df.columns)[:max_periods]
    periods = [str(getattr(c, "year", str(c)[:4])) for c in cols]
    out_rows: dict[str, list] = {}
    for r in rows:
        if r in df.index:
            vals = []
            for c in cols:
                vals.append(_f(df.loc[r, c]))
            out_rows[r] = vals
    if not out_rows:
        return None
    return {"periods": periods, "rows": out_rows}


def fetch_statements(ticker: str, use_cache: bool = True) -> dict:
    """Income / balance / cash-flow history (annual) + analyst targets and
    recommendation mix for a single ticker. Returns {} if unavailable."""
    ticker = str(ticker).upper()
    cache_name = f"statements_{ticker}.json"
    if use_cache:
        cached = _load_cache(cache_name, 24 * 60 * 60)
        if cached is not None:
            return cached

    data: dict = {}
    tk = yf.Ticker(ticker) if yf is not None else None
    if tk is not None:
        try:
            data["income"] = _statement_dict(tk.income_stmt, _INCOME_ROWS)
        except Exception:
            pass
        try:
            data["balance"] = _statement_dict(tk.balance_sheet, _BALANCE_ROWS)
        except Exception:
            pass
        try:
            data["cashflow"] = _statement_dict(tk.cashflow, _CASHFLOW_ROWS)
        except Exception:
            pass
        try:
            pt = tk.analyst_price_targets
            if isinstance(pt, dict) and pt:
                data["target"] = {k: _f(v) for k, v in pt.items()}
        except Exception:
            pass
        try:
            rs = tk.recommendations_summary
            if rs is not None and len(rs):
                row = rs.iloc[0].to_dict()
                data["recommendation"] = {
                    k: int(row[k]) for k in
                    ("strongBuy", "buy", "hold", "sell", "strongSell")
                    if k in row and row[k] == row[k]
                }
        except Exception:
            pass

    if use_cache and any(data.get(k) for k in ("income", "balance", "cashflow")):
        _save_cache(cache_name, data)
    return data
