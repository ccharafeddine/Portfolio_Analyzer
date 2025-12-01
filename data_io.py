import os
from typing import List, Optional, Dict

import requests
import pandas as pd
from datetime import datetime


# -------------------------
# Config
# -------------------------

FMP_BASE_URL = "https://financialmodelingprep.com/api/v3"

# Mapping from crypto tickers used in your app to CoinGecko IDs
CRYPTO_TICKER_MAP: Dict[str, str] = {
    "BTC-USD": "bitcoin",
    # add more here if you ever need them, e.g.:
    # "ETH-USD": "ethereum",
}


# -------------------------
# Helper: Get FMP API key
# -------------------------

def _get_fmp_api_key() -> str:
    """
    Get the FMP API key from either environment variables or (if running
    under Streamlit) from st.secrets["FMP_API_KEY"].
    """
    key = os.getenv("FMP_API_KEY")
    if key:
        return key

    # Try Streamlit secrets (for Streamlit Cloud deployments)
    try:
        import streamlit as st  # type: ignore

        if "FMP_API_KEY" in st.secrets:
            return st.secrets["FMP_API_KEY"]
    except Exception:
        pass

    raise RuntimeError(
        "FMP_API_KEY not found. Set it as an environment variable or in "
        "Streamlit secrets."
    )


# -------------------------
# FMP price + dividend helpers
# -------------------------

def _fetch_fmp_prices_for_ticker(
    ticker: str, start: str, end: Optional[str]
) -> pd.Series:
    """
    Fetch historical adjusted close prices for a single ticker from FMP.

    Returns a Series indexed by Date with the column name = ticker.
    If FMP returns 403/429, we log a warning and return an empty Series
    so the caller can skip this ticker.
    """
    api_key = _get_fmp_api_key()
    params: Dict[str, str] = {"apikey": api_key}
    if start:
        params["from"] = start
    if end:
        params["to"] = end

    url = f"{FMP_BASE_URL}/historical-price-full/{ticker}"
    resp = requests.get(url, params=params, timeout=15)

    if resp.status_code in (403, 429):
        print(
            f"Warning: FMP returned {resp.status_code} for {ticker}. "
            "This usually means the symbol is not available on your plan "
            "or you hit a temporary limit. Skipping this ticker."
        )
        return pd.Series(dtype="float64", name=ticker)

    resp.raise_for_status()
    data = resp.json()

    hist = data.get("historical", [])
    if not hist:
        return pd.Series(dtype="float64", name=ticker)

    df = pd.DataFrame(hist)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()

    # Prefer adjClose if available, else close
    if "adjClose" in df.columns:
        s = df["adjClose"].astype(float)
    else:
        s = df["close"].astype(float)

    s.name = ticker
    s.index.name = "Date"
    return s


def _fetch_fmp_dividends_for_ticker(
    ticker: str, start: str, end: Optional[str]
) -> pd.Series:
    """
    Fetch historical cash dividends for a single stock/ETF from FMP.

    Returns a Series indexed by Date with dividend amounts.
    If no dividend data is available (or not supported), returns empty Series.
    """
    api_key = _get_fmp_api_key()
    params: Dict[str, str] = {"apikey": api_key}
    if start:
        params["from"] = start
    if end:
        params["to"] = end

    url = f"{FMP_BASE_URL}/historical-price-full/stock_dividend/{ticker}"
    resp = requests.get(url, params=params, timeout=15)

    if resp.status_code in (403, 429):
        print(
            f"Warning: FMP returned {resp.status_code} for dividends of {ticker}. "
            "Skipping dividends for this ticker."
        )
        return pd.Series(dtype="float64", name=ticker)

    if resp.status_code == 404:
        # No dividend endpoint for this symbol
        return pd.Series(dtype="float64", name=ticker)

    resp.raise_for_status()
    data = resp.json()
    hist = data.get("historical", [])
    if not hist:
        return pd.Series(dtype="float64", name=ticker)

    df = pd.DataFrame(hist)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()

    col = "adjDividend" if "adjDividend" in df.columns else "dividend"
    s = df[col].astype(float)
    s.name = ticker
    s.index.name = "Date"
    return s


# -------------------------
# CoinGecko helper for BTC-USD (and other crypto)
# -------------------------

def _fetch_crypto_prices_from_coingecko(
    ticker: str, start: str, end: Optional[str]
) -> pd.Series:
    """
    Fetch daily close prices in USD for a crypto ticker using CoinGecko.

    Currently supports mapping in CRYPTO_TICKER_MAP (e.g. BTC-USD -> bitcoin).

    We use the 'market_chart' endpoint and convert timestamps to daily closes.
    """
    ticker_upper = ticker.upper()
    if ticker_upper not in CRYPTO_TICKER_MAP:
        # Not a known crypto ticker for this helper
        return pd.Series(dtype="float64", name=ticker)

    cg_id = CRYPTO_TICKER_MAP[ticker_upper]

    # Parse dates and compute range in days
    start_dt = datetime.fromisoformat(str(start)).date()
    if end is not None:
        end_dt = datetime.fromisoformat(str(end)).date()
    else:
        end_dt = datetime.utcnow().date()

    days_span = (end_dt - start_dt).days
    if days_span < 1:
        days_span = 1

    # CoinGecko only allows integer 'days' up to some max; we'll request a bit more
    days_param = days_span + 5

    url = (
        f"https://api.coingecko.com/api/v3/coins/{cg_id}/market_chart"
        f"?vs_currency=usd&days={days_param}"
    )

    resp = requests.get(url, timeout=20)
    if resp.status_code in (403, 429):
        print(
            f"Warning: CoinGecko returned {resp.status_code} for {ticker}. "
            "Crypto data may be temporarily rate-limited."
        )
        return pd.Series(dtype="float64", name=ticker)

    resp.raise_for_status()
    data = resp.json()

    # data["prices"] is list of [timestamp_ms, price]
    prices_list = data.get("prices", [])
    if not prices_list:
        return pd.Series(dtype="float64", name=ticker)

    df = pd.DataFrame(prices_list, columns=["ts", "price"])
    df["Date"] = pd.to_datetime(df["ts"], unit="ms").dt.date
    # Take last price per day as daily close
    daily = df.groupby("Date")["price"].last()
    daily.index = pd.to_datetime(daily.index)
    daily = daily.sort_index()

    # Filter to requested range
    mask = (daily.index.date >= start_dt) & (daily.index.date <= end_dt)
    daily = daily.loc[mask]

    s = daily.astype(float)
    s.name = ticker
    s.index.name = "Date"
    return s


# -------------------------
# Public API used by main.py
# -------------------------

def download_prices(
    tickers: List[str], start: str, end: Optional[str], interval: str = "1d"
) -> pd.DataFrame:
    """
    Download adjusted close prices for a list of tickers.

    - For known crypto tickers (e.g. BTC-USD), use CoinGecko.
    - For all others, use FMP.
    - Symbols that cannot be fetched are skipped with a warning.
    """
    series_list = []

    for t in tickers:
        t_upper = t.upper()

        # Crypto via CoinGecko
        if t_upper in CRYPTO_TICKER_MAP:
            s = _fetch_crypto_prices_from_coingecko(t_upper, start, end)
            if s.empty:
                print(f"Warning: no CoinGecko price data for {t_upper} in the requested period.")
            else:
                series_list.append(s)
            continue

        # Everything else via FMP
        s = _fetch_fmp_prices_for_ticker(t_upper, start, end)
        if s.empty:
            print(f"Warning: no FMP price data for {t_upper} in the requested period.")
        else:
            series_list.append(s)

    if not series_list:
        raise ValueError(
            "No price data returned from FMP/CoinGecko for any ticker. "
            "Check your tickers, dates, API key, or network connectivity."
        )

    prices = pd.concat(series_list, axis=1)
    prices.index.name = "Date"
    return prices.sort_index()


def download_dividends(
    tickers: List[str], start: str, end: Optional[str]
) -> pd.DataFrame:
    """
    Download dividend cash flows for tickers.

    - Crypto tickers (e.g. BTC-USD) are assumed to have no dividends.
    - Stocks/ETFs use FMP's stock_dividend endpoint.
    """
    divs: Dict[str, pd.Series] = {}

    for t in tickers:
        t_upper = t.upper()

        # Crypto: no dividends
        if t_upper in CRYPTO_TICKER_MAP:
            continue

        s = _fetch_fmp_dividends_for_ticker(t_upper, start, end)
        if s.empty:
            continue
        divs[t_upper] = s

    if not divs:
        return pd.DataFrame()

    div_df = pd.concat(divs, axis=1).fillna(0.0)
    div_df.index.name = "Date"
    return div_df.sort_index()


def monthly_returns_from_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute monthly close-to-close returns from daily prices.

    Uses month-end prices, then pct_change with fill_method=None to avoid
    the deprecated default.
    """
    monthly = prices.resample("ME").last()
    rets = monthly.pct_change(fill_method=None).dropna(how="all")
    return rets


def ensure_output_dir(path: str = "outputs") -> str:
    os.makedirs(path, exist_ok=True)
    return path
