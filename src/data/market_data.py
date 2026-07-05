"""Market context for the News & Macro tab: per-holding news and macro/rates.

Two zero-key baselines with optional keyed enhancements:

- **News** — yfinance headlines per ticker (no key). If an Alpha Vantage key is
  supplied, also pull the ``NEWS_SENTIMENT`` feed (more articles + per-article
  sentiment) and merge/dedupe with the yfinance items.
- **Macro** — requires a free FRED key; returns ``None`` without one (the UI shows a
  prompt). Pulls the Treasury curve plus a few headline rates via the FRED REST API.

This module is Qt-free so it can run headless and be unit-tested. Every network call
is defensive: a failure returns baseline/empty data rather than raising. Responses are
cached to disk with a short TTL, which also protects the tight Alpha Vantage free-tier
quota (25 requests/day).
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import quote

import pandas as pd

try:
    import requests
except Exception:  # pragma: no cover - requests is a declared dependency
    requests = None

try:
    import yfinance as yf
except Exception:  # pragma: no cover - yfinance is a declared dependency
    yf = None


# ── Cache (shares the app cache dir; falls back to home like fetcher.py) ──
_CACHE_DIR = Path(
    os.getenv("PORTFOLIO_ANALYZER_CACHE_DIR")
    or (Path.home() / ".portfolio_analyzer_cache")
).parent / "market"

CACHE_TTL_NEWS = 15 * 60        # 15 minutes
CACHE_TTL_MACRO = 6 * 60 * 60   # 6 hours

_TAG_RE = re.compile(r"<[^>]+>")


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


def _strip_html(text: Optional[str]) -> str:
    if not text:
        return ""
    return _TAG_RE.sub("", text).strip()


def _to_float(v) -> Optional[float]:
    try:
        f = float(v)
        return f if f == f else None  # drop NaN
    except (TypeError, ValueError):
        return None


def _parse_dt(v) -> Optional[datetime]:
    """Parse the various date formats these APIs return, always as UTC-aware."""
    if v is None or v == "":
        return None
    # Epoch seconds (older yfinance)
    if isinstance(v, (int, float)):
        try:
            return datetime.fromtimestamp(v, tz=timezone.utc)
        except Exception:
            return None
    s = str(v)
    # Alpha Vantage: "20260701T211000"
    if re.fullmatch(r"\d{8}T\d{6}", s):
        try:
            return datetime.strptime(s, "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
        except Exception:
            return None
    # ISO 8601, possibly with a trailing Z
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────
# News
# ──────────────────────────────────────────────────────────────

@dataclass
class NewsItem:
    title: str
    publisher: str = ""
    url: str = ""
    published: Optional[datetime] = None
    summary: str = ""
    tickers: list = field(default_factory=list)
    sentiment: Optional[float] = None
    sentiment_label: Optional[str] = None
    thumbnail: Optional[str] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["published"] = self.published.isoformat() if self.published else None
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "NewsItem":
        d = dict(d)
        d["published"] = _parse_dt(d.get("published"))
        return cls(**d)


def _parse_yf_news(raw, ticker: str) -> list[NewsItem]:
    out: list[NewsItem] = []
    for item in raw or []:
        # yfinance >=0.2.40 nests fields under "content"; older versions are flat.
        c = item.get("content") if isinstance(item, dict) and "content" in item else item
        if not isinstance(c, dict):
            continue
        title = c.get("title")
        if not title:
            continue
        provider = c.get("provider") or {}
        publisher = provider.get("displayName") if isinstance(provider, dict) else c.get("publisher", "")
        url = ""
        for key in ("clickThroughUrl", "canonicalUrl"):
            ref = c.get(key)
            if isinstance(ref, dict) and ref.get("url"):
                url = ref["url"]
                break
        if not url:
            url = c.get("link", "")
        published = _parse_dt(c.get("pubDate") or c.get("displayTime") or c.get("providerPublishTime"))
        summary = _strip_html(c.get("summary") or c.get("description"))
        thumb = None
        tn = c.get("thumbnail")
        if isinstance(tn, dict):
            thumb = tn.get("originalUrl") or (tn.get("resolutions") or [{}])[0].get("url")
        out.append(NewsItem(
            title=title, publisher=publisher or "", url=url, published=published,
            summary=summary, tickers=[ticker], thumbnail=thumb,
        ))
    return out


def _fetch_av_news(tickers: list[str], key: str, limit: int = 50) -> list[NewsItem]:
    if not (requests and key):
        return []
    resp = requests.get(
        "https://www.alphavantage.co/query",
        params={
            "function": "NEWS_SENTIMENT",
            "tickers": ",".join(tickers),
            "apikey": key,
            "limit": limit,
            "sort": "LATEST",
        },
        timeout=15,
    )
    data = resp.json()
    out: list[NewsItem] = []
    for a in data.get("feed", []) or []:
        rel = [
            ts.get("ticker") for ts in a.get("ticker_sentiment", []) or []
            if ts.get("ticker") in tickers
        ]
        out.append(NewsItem(
            title=a.get("title", ""),
            publisher=a.get("source", ""),
            url=a.get("url", ""),
            published=_parse_dt(a.get("time_published")),
            summary=_strip_html(a.get("summary")),
            tickers=rel,
            sentiment=_to_float(a.get("overall_sentiment_score")),
            sentiment_label=a.get("overall_sentiment_label"),
            thumbnail=a.get("banner_image") or None,
        ))
    return out


def _dedupe(items: list[NewsItem]) -> list[NewsItem]:
    seen: dict[str, NewsItem] = {}
    for it in items:
        key = (it.url.split("?")[0].rstrip("/").lower() if it.url else it.title.strip().lower())
        if not key:
            continue
        prev = seen.get(key)
        # Prefer the copy that carries sentiment; otherwise keep the first seen.
        if prev is None:
            seen[key] = it
        elif prev.sentiment is None and it.sentiment is not None:
            it.tickers = sorted(set(prev.tickers) | set(it.tickers))
            seen[key] = it
        else:
            prev.tickers = sorted(set(prev.tickers) | set(it.tickers))
    return list(seen.values())


def fetch_ticker_news(
    tickers, av_key: Optional[str] = None, per_ticker: int = 8,
    total_cap: int = 40, use_cache: bool = True,
) -> list[NewsItem]:
    """Per-holding news: yfinance baseline plus optional Alpha Vantage sentiment."""
    tickers = [str(t).upper() for t in tickers if t]
    if not tickers:
        return []
    cache_name = f"news_{'-'.join(sorted(tickers))}_{'av' if av_key else 'yf'}.json"
    if use_cache:
        cached = _load_cache(cache_name, CACHE_TTL_NEWS)
        if cached is not None:
            return [NewsItem.from_dict(d) for d in cached]

    items: list[NewsItem] = []
    if yf is not None:
        for t in tickers:
            try:
                items += _parse_yf_news(yf.Ticker(t).news, t)[:per_ticker]
            except Exception:
                continue
    if av_key:
        try:
            items += _fetch_av_news(tickers, av_key)
        except Exception:
            pass

    items = _dedupe(items)
    _epoch = datetime.min.replace(tzinfo=timezone.utc)
    items.sort(key=lambda n: n.published or _epoch, reverse=True)
    items = items[:total_cap]
    if use_cache and items:
        _save_cache(cache_name, [n.to_dict() for n in items])
    return items


# ──────────────────────────────────────────────────────────────
# Macro (FRED)
# ──────────────────────────────────────────────────────────────

# series_id -> display name
FRED_SERIES = {
    "DGS3MO": "3-Month Treasury",
    "DGS2": "2-Year Treasury",
    "DGS10": "10-Year Treasury",
    "DGS30": "30-Year Treasury",
    "FEDFUNDS": "Fed Funds Rate",
    "UNRATE": "Unemployment Rate",
    "CPIAUCSL": "CPI (YoY)",
}
# series that form the yield curve, in tenor order
_CURVE_TENORS = [("DGS3MO", "3M"), ("DGS2", "2Y"), ("DGS10", "10Y"), ("DGS30", "30Y")]
_HISTORY_SERIES = ["DGS10", "FEDFUNDS"]


@dataclass
class MacroData:
    curve: dict = field(default_factory=dict)      # tenor -> yield (%)
    rates: list = field(default_factory=list)      # {name, value, date, change_1y}
    series: dict = field(default_factory=dict)     # series_id -> pd.Series
    as_of: Optional[datetime] = None

    def to_dict(self) -> dict:
        return {
            "curve": self.curve,
            "rates": self.rates,
            "series": {
                sid: {ts.isoformat(): float(v) for ts, v in s.items()}
                for sid, s in self.series.items()
            },
            "as_of": self.as_of.isoformat() if self.as_of else None,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MacroData":
        series = {}
        for sid, obj in (d.get("series") or {}).items():
            try:
                series[sid] = pd.Series(
                    {pd.Timestamp(k): float(v) for k, v in obj.items()}
                ).sort_index()
            except Exception:
                continue
        return cls(
            curve=d.get("curve") or {},
            rates=d.get("rates") or [],
            series=series,
            as_of=_parse_dt(d.get("as_of")),
        )


def _fred_series(series_id: str, key: str, limit: int = 520) -> Optional[pd.Series]:
    if not (requests and key):
        return None
    resp = requests.get(
        "https://api.stlouisfed.org/fred/series/observations",
        params={
            "series_id": series_id,
            "api_key": key,
            "file_type": "json",
            "sort_order": "desc",
            "limit": limit,
        },
        timeout=15,
    )
    data = resp.json()
    rows = []
    for o in data.get("observations", []) or []:
        val = o.get("value")
        if val in (".", "", None):
            continue
        try:
            rows.append((pd.Timestamp(o["date"]), float(val)))
        except Exception:
            continue
    if not rows:
        return None
    return pd.Series(dict(rows)).sort_index()


def _change_1y(s: pd.Series) -> Optional[float]:
    try:
        cutoff = s.index[-1] - pd.Timedelta(days=365)
        past = s[s.index <= cutoff]
        if past.empty:
            return None
        return float(s.iloc[-1]) - float(past.iloc[-1])
    except Exception:
        return None


def fetch_macro(fred_key: Optional[str], use_cache: bool = True) -> Optional[MacroData]:
    """Treasury curve + headline rates via FRED. Returns None without a key/data."""
    if not (requests and fred_key):
        return None
    if use_cache:
        cached = _load_cache("macro_fred.json", CACHE_TTL_MACRO)
        if cached is not None:
            return MacroData.from_dict(cached)

    series: dict[str, pd.Series] = {}
    for sid in FRED_SERIES:
        s = _fred_series(sid, fred_key)
        if s is not None and not s.empty:
            series[sid] = s
    if not series:
        return None

    curve = {ten: float(series[sid].iloc[-1]) for sid, ten in _CURVE_TENORS if sid in series}

    rates = []
    for sid, name in FRED_SERIES.items():
        if sid not in series:
            continue
        s = series[sid]
        latest = float(s.iloc[-1])
        change = _change_1y(s)
        if sid == "CPIAUCSL":  # convert the index level to a YoY %
            try:
                cutoff = s.index[-1] - pd.Timedelta(days=365)
                prior = s[s.index <= cutoff]
                latest = (latest / float(prior.iloc[-1]) - 1.0) * 100.0 if not prior.empty else latest
                change = None
            except Exception:
                change = None
        rates.append({
            "name": name,
            "value": latest,
            "date": s.index[-1].date().isoformat(),
            "change_1y": change,
        })

    history = {}
    for sid in _HISTORY_SERIES:
        if sid in series:
            history[sid] = series[sid].tail(1300)  # ~5y of business days

    macro = MacroData(curve=curve, rates=rates, series=history,
                      as_of=datetime.now(timezone.utc))
    if use_cache:
        _save_cache("macro_fred.json", macro.to_dict())
    return macro


# ──────────────────────────────────────────────────────────────
# Upcoming events calendar (earnings + ex-dividend)
# ──────────────────────────────────────────────────────────────

@dataclass
class CalendarEvent:
    ticker: str
    date: str      # ISO date (YYYY-MM-DD)
    kind: str      # "Earnings" | "Ex-Dividend"
    detail: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "CalendarEvent":
        return cls(**d)


def _as_date(v) -> Optional[date]:
    if v is None:
        return None
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, date):
        return v
    try:
        return datetime.fromisoformat(str(v)[:10]).date()
    except Exception:
        return None


def fetch_calendar(tickers, use_cache: bool = True) -> list[CalendarEvent]:
    """Upcoming earnings and ex-dividend dates across the holdings, soonest first.
    Uses only ``Ticker.calendar`` (cheap), so it's lighter than full fundamentals."""
    tickers = [str(t).upper() for t in tickers if t]
    if not tickers:
        return []
    cache_name = f"calendar_{'-'.join(sorted(tickers))}.json"
    if use_cache:
        cached = _load_cache(cache_name, CACHE_TTL_MACRO)
        if cached is not None:
            return [CalendarEvent.from_dict(d) for d in cached]

    today = datetime.now(timezone.utc).date()
    events: list[CalendarEvent] = []
    if yf is not None:
        for t in tickers:
            try:
                cal = yf.Ticker(t).calendar or {}
            except Exception:
                continue
            avg = cal.get("Earnings Average")
            detail = f"Est. EPS ${avg:.2f}" if isinstance(avg, (int, float)) else ""
            ed = cal.get("Earnings Date")
            for d in (ed if isinstance(ed, (list, tuple)) else [ed]):
                dd = _as_date(d)
                if dd and dd >= today:
                    events.append(CalendarEvent(t, dd.isoformat(), "Earnings", detail))
            xd = _as_date(cal.get("Ex-Dividend Date"))
            if xd and xd >= today:
                events.append(CalendarEvent(t, xd.isoformat(), "Ex-Dividend", ""))

    events.sort(key=lambda e: e.date)
    if use_cache and events:
        _save_cache(cache_name, [e.to_dict() for e in events])
    return events


# ──────────────────────────────────────────────────────────────
# Live quotes (delayed) — powers the ticker strip + Live Market Watch
# ──────────────────────────────────────────────────────────────
#
# yfinance does not stream; ``fast_info`` gives a lightweight delayed snapshot
# (typically 15–20 min behind the exchange). We poll it on an interval from the
# UI. A tiny in-memory TTL cache dedupes bursts so a strip + view asking at the
# same moment hit the network once.

CACHE_TTL_QUOTES = 10  # seconds — just enough to dedupe near-simultaneous polls


# ── Symbol normalization ─────────────────────────────────────────
# Curated crypto shorthand -> yfinance's canonical dashed pair. yfinance prices
# crypto as "<COIN>-USD" (e.g. BTC-USD). This is deliberately a CURATED allow-list,
# NOT an "append -USD" heuristic: bare tickers like BTC collide with real equities
# (BTC is Grayscale's Bitcoin Mini Trust ETF), so we only remap coins we explicitly
# recognize and pass everything else through untouched. Kept as a plain dict so a
# central symbol resolver can later replace this without changing any call site.
_CRYPTO_BASES = [
    "BTC", "ETH", "SOL", "XRP", "ADA", "DOGE", "DOT", "LTC", "BCH", "LINK",
    "AVAX", "MATIC", "UNI", "ATOM", "XLM", "ETC", "ALGO", "TRX", "SHIB", "XMR",
]
_CRYPTO_ALIASES: dict[str, str] = {}
for _base in _CRYPTO_BASES:
    _canonical = f"{_base}-USD"
    _CRYPTO_ALIASES[_base] = _canonical            # BTC    -> BTC-USD
    _CRYPTO_ALIASES[f"{_base}USD"] = _canonical    # BTCUSD -> BTC-USD

_WS_RE = re.compile(r"\s+")


def normalize_symbol(raw) -> str:
    """Clean and canonicalize a user-typed symbol for yfinance.

    Strips/uppercases, maps a curated set of crypto shorthands to their dashed
    pair (BTC/BTCUSD -> BTC-USD), and passes valid equity/ETF/crypto symbols
    through unchanged. Returns "" for empty or junk input (no alphanumerics),
    which callers treat as unresolvable.
    """
    s = _WS_RE.sub("", str(raw or "").upper())
    if not any(ch.isalnum() for ch in s):
        return ""
    return _CRYPTO_ALIASES.get(s, s)


@dataclass
class Quote:
    ticker: str
    last: Optional[float] = None
    prev_close: Optional[float] = None
    change: Optional[float] = None      # last - prev_close
    change_pct: Optional[float] = None  # change / prev_close (fraction, e.g. 0.012)
    day_high: Optional[float] = None
    day_low: Optional[float] = None
    volume: Optional[float] = None
    currency: Optional[str] = None
    name: Optional[str] = None          # best-effort human-readable name
    as_of: Optional[str] = None         # ISO timestamp of the fetch
    source: str = "yfinance"            # data source that produced this quote
    realtime: bool = False              # True for a real-time provider feed
    error: Optional[str] = None         # set when the per-symbol fetch failed

    @property
    def ok(self) -> bool:
        return self.last is not None

    def _fill_change(self) -> None:
        """Derive change / change_pct from last + prev_close when possible."""
        if self.last is not None and self.prev_close not in (None, 0):
            self.change = self.last - self.prev_close
            self.change_pct = (self.last - self.prev_close) / self.prev_close


# frozenset(tickers) -> (monotonic_ts, {ticker: Quote})
_QUOTE_CACHE: dict[frozenset, tuple[float, dict]] = {}


def _f(v) -> Optional[float]:
    """Coerce to a finite float or None."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f != f or f in (float("inf"), float("-inf")):  # NaN / inf guard
        return None
    return f


def fetch_quotes(
    tickers, use_cache: bool = True, provider=None, creds=None, with_names: bool = False
) -> dict[str, Quote]:
    """Return a snapshot quote per ticker.

    Default source is yfinance ``fast_info`` (delayed). When ``provider`` is one
    of ``finnhub`` / ``polygon`` / ``alpaca`` and ``creds`` is set, quotes come
    from that real-time feed instead; any provider failure falls back to
    yfinance so the strip never goes dark. Best-effort and Qt-free: a per-ticker
    failure yields an empty ``Quote`` (``ok`` False) rather than raising.

    ``with_names=True`` additionally fills a best-effort human-readable name per
    symbol (heavier ``.info``, long-TTL cached). Only the Live Market Watch
    watchlist asks for names; the always-on ticker strip stays on the cheap path.
    """
    syms = [str(t).strip().upper() for t in (tickers or []) if str(t).strip()]
    syms = list(dict.fromkeys(syms))  # dedupe, preserve order
    if not syms:
        return {}

    cache_key = (provider or "yf", frozenset(syms))
    out: Optional[dict[str, Quote]] = None
    if use_cache:
        hit = _QUOTE_CACHE.get(cache_key)
        if hit and (time.monotonic() - hit[0]) < CACHE_TTL_QUOTES:
            out = dict(hit[1])

    if out is None:
        out = {}
        if provider:
            try:
                out = _provider_quotes(provider, syms, creds)
            except Exception:
                out = {}
        # Fill any gaps (or everything, if no provider) from yfinance-delayed.
        missing = [t for t in syms if t not in out or not out[t].ok]
        if missing:
            out.update(_yfinance_quotes(missing))
        _QUOTE_CACHE[cache_key] = (time.monotonic(), dict(out))

    if with_names:
        for sym, q in out.items():
            if q.ok and not q.name:
                q.name = _yf_name(sym)
    return out


def _yfinance_quotes(syms) -> dict[str, Quote]:
    now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if yf is None:
        return {t: Quote(t, as_of=now_iso) for t in syms}
    out: dict[str, Quote] = {}
    for t in syms:
        q = Quote(t, as_of=now_iso, source="yfinance", realtime=False)
        try:
            fi = yf.Ticker(t).fast_info
            q.last = _f(_fi_get(fi, "last_price"))
            q.prev_close = _f(_fi_get(fi, "previous_close"))
            q._fill_change()
            q.day_high = _f(_fi_get(fi, "day_high"))
            q.day_low = _f(_fi_get(fi, "day_low"))
            q.volume = _f(_fi_get(fi, "last_volume"))
            cur = _fi_get(fi, "currency")
            q.currency = str(cur) if cur else None
        except Exception as e:
            q.error = str(e)  # leave q as an empty (not-ok) quote, flagged
        out[t] = q
    return out


# ── Best-effort names (Live Market Watch watchlist) ──────────────
# Names come from yfinance's heavier ``.info``; cached long since they rarely
# change. Only the watchlist requests them (``with_names=True``) so the always-on
# ticker strip stays on the cheap ``fast_info`` path.
CACHE_TTL_NAMES = 24 * 60 * 60     # 24h for a resolved name
CACHE_TTL_NAMES_MISS = 10 * 60     # 10m before retrying an unresolved one
# symbol -> (monotonic_ts, name|None)
_NAME_CACHE: dict[str, tuple[float, Optional[str]]] = {}


def _yf_name(sym: str) -> Optional[str]:
    """Best-effort display name for ``sym`` (long-TTL cached, never raises)."""
    hit = _NAME_CACHE.get(sym)
    if hit is not None:
        ttl = CACHE_TTL_NAMES if hit[1] else CACHE_TTL_NAMES_MISS
        if (time.monotonic() - hit[0]) < ttl:
            return hit[1]
    name: Optional[str] = None
    if yf is not None:
        try:
            info = yf.Ticker(sym).info or {}
            raw = info.get("shortName") or info.get("longName")
            name = (str(raw).strip() or None) if raw else None
        except Exception:
            name = None
    _NAME_CACHE[sym] = (time.monotonic(), name)
    return name


# ── Real-time providers ──────────────────────────────────────────
# Each returns {ticker: Quote} for the symbols it could price. Best-effort:
# raise or return partial; fetch_quotes fills the rest from yfinance.

def _provider_quotes(provider: str, syms, creds) -> dict[str, Quote]:
    if provider == "finnhub":
        return _finnhub_quotes(syms, creds)
    if provider == "polygon":
        return _polygon_quotes(syms, creds)
    if provider == "alpaca":
        return _alpaca_quotes(syms, creds)
    return {}


def _http_get_json(url: str, params=None, headers=None, timeout: int = 6):
    if requests is None:
        return None
    # Fail closed: a requests exception message embeds the full URL, which would
    # leak the provider API key (token=/apiKey=) into any surfaced error string.
    try:
        r = requests.get(url, params=params, headers=headers, timeout=timeout)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None


def _finnhub_quotes(syms, key) -> dict[str, Quote]:
    """Finnhub /quote: {c: current, pc: prev close, h, l, o, t: unix time}."""
    now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
    out: dict[str, Quote] = {}
    for t in syms:
        data = _http_get_json("https://finnhub.io/api/v1/quote",
                              params={"symbol": t, "token": key})
        if not data:
            continue
        last = _f(data.get("c"))
        if not last:  # 0 / None means Finnhub had nothing
            continue
        q = Quote(t, as_of=now_iso, source="finnhub", realtime=True)
        q.last = last
        q.prev_close = _f(data.get("pc"))
        q.day_high = _f(data.get("h"))
        q.day_low = _f(data.get("l"))
        q._fill_change()
        out[t] = q
    return out


def _polygon_quotes(syms, key) -> dict[str, Quote]:
    """Polygon snapshot: ticker.day.c (last), ticker.prevDay.c (prev close)."""
    now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
    out: dict[str, Quote] = {}
    for t in syms:
        data = _http_get_json(
            "https://api.polygon.io/v2/snapshot/locale/us/markets/stocks/tickers/"
            f"{quote(t, safe='')}",
            params={"apiKey": key},
        )
        tk = (data or {}).get("ticker") or {}
        day = tk.get("day") or {}
        prev = tk.get("prevDay") or {}
        last_trade = (tk.get("lastTrade") or {}).get("p")
        last = _f(last_trade) or _f(day.get("c"))
        if not last:
            continue
        q = Quote(t, as_of=now_iso, source="polygon", realtime=True)
        q.last = last
        q.prev_close = _f(prev.get("c"))
        q.day_high = _f(day.get("h"))
        q.day_low = _f(day.get("l"))
        q.volume = _f(day.get("v"))
        q._fill_change()
        out[t] = q
    return out


def _alpaca_quotes(syms, creds) -> dict[str, Quote]:
    """Alpaca latest trades (IEX feed). ``creds`` is ``(key, secret)``."""
    if not isinstance(creds, (tuple, list)) or len(creds) != 2:
        return {}
    key, secret = creds
    now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
    headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
    data = _http_get_json(
        "https://data.alpaca.markets/v2/stocks/trades/latest",
        params={"symbols": ",".join(syms), "feed": "iex"}, headers=headers,
    )
    trades = (data or {}).get("trades") or {}
    if not trades:
        return {}
    # Previous closes come from yfinance (Alpaca's bar endpoint needs a second
    # call); fetch_quotes fills prev_close-derived change via the fallback path
    # for anything left incomplete. Here we at least deliver the real-time last.
    out: dict[str, Quote] = {}
    for t in syms:
        tr = trades.get(t) or {}
        last = _f(tr.get("p"))
        if not last:
            continue
        q = Quote(t, as_of=now_iso, source="alpaca", realtime=True)
        q.last = last
        out[t] = q
    return out


def _fi_get(fast_info, name):
    """``fast_info`` supports both attribute and mapping access across yfinance
    versions; try attribute first, then mapping, then give up."""
    try:
        val = getattr(fast_info, name)
        if val is not None:
            return val
    except Exception:
        pass
    try:
        return fast_info[name]
    except Exception:
        return None


def fetch_intraday(ticker: str, use_cache: bool = True):
    """Return today's 1-minute OHLCV frame for one ticker (or ``None``).

    Best-effort: any failure (no data, network, delisted) returns ``None`` so
    the Live Market Watch chart just stays blank. Powers the click-through
    intraday chart. A short in-memory TTL cache avoids re-fetching the same
    symbol on rapid row clicks.
    """
    sym = str(ticker or "").strip().upper()
    if not sym or yf is None:
        return None

    if use_cache:
        hit = _INTRADAY_CACHE.get(sym)
        if hit and (time.monotonic() - hit[0]) < CACHE_TTL_INTRADAY:
            return hit[1]

    try:
        df = yf.Ticker(sym).history(period="1d", interval="1m")
    except Exception:
        return None
    if df is None or getattr(df, "empty", True):
        return None
    _INTRADAY_CACHE[sym] = (time.monotonic(), df)
    return df


CACHE_TTL_INTRADAY = 60  # seconds
# symbol -> (monotonic_ts, DataFrame)
_INTRADAY_CACHE: dict[str, tuple[float, object]] = {}


# ── Multi-timeframe OHLC (TradingView-style price chart) ──────────
# The Live Market Watch price chart offers TradingView-style timeframe buttons.
# Each maps to a yfinance (period, interval) pair. Sub-daily intervals are capped
# by yfinance to short lookbacks (1m<=7d, 5m/30m<=60d), which the periods respect.
OHLC_TIMEFRAMES: dict[str, tuple[str, str]] = {
    "1D": ("1d", "1m"),
    "5D": ("5d", "5m"),
    "1M": ("1mo", "30m"),
    "6M": ("6mo", "1d"),
    "1Y": ("1y", "1d"),
    "5Y": ("5y", "1wk"),
}
DEFAULT_TIMEFRAME = "1D"
# A timeframe is "intraday" (sub-daily bars → timestamped x-axis) vs daily+.
INTRADAY_TIMEFRAMES = {"1D", "5D", "1M"}

CACHE_TTL_OHLC = 30  # seconds — dedupe rapid selection/timeframe clicks
# (symbol, timeframe) -> (monotonic_ts, DataFrame)
_OHLC_CACHE: dict[tuple[str, str], tuple[float, object]] = {}


def fetch_ohlc(ticker: str, timeframe: str = DEFAULT_TIMEFRAME, use_cache: bool = True):
    """Return an OHLCV frame for one ticker at a TradingView-style ``timeframe``
    (one of :data:`OHLC_TIMEFRAMES`), or ``None``.

    Best-effort and Qt-free: any failure (no data, network, delisted, unknown
    timeframe) returns ``None`` so the chart just stays blank. A short in-memory
    TTL cache dedupes the bursts from rapid row / timeframe clicks. Powers the
    Live Market Watch candlestick chart (fed OHLCV, not just a close line).
    """
    sym = str(ticker or "").strip().upper()
    tf = timeframe if timeframe in OHLC_TIMEFRAMES else DEFAULT_TIMEFRAME
    if not sym or yf is None:
        return None

    key = (sym, tf)
    if use_cache:
        hit = _OHLC_CACHE.get(key)
        if hit and (time.monotonic() - hit[0]) < CACHE_TTL_OHLC:
            return hit[1]

    period, interval = OHLC_TIMEFRAMES[tf]
    try:
        df = yf.Ticker(sym).history(period=period, interval=interval)
    except Exception:
        return None
    if df is None or getattr(df, "empty", True):
        return None
    _OHLC_CACHE[key] = (time.monotonic(), df)
    return df
