import os
from typing import List, Optional, Dict

import requests
import pandas as pd
from datetime import datetime


# -------------------------
# Config
# -------------------------

FMP_BASE_URL = "https://financialmodelingprep.com/api/v3"

# Mapping from crypto tickers used in your app to Alpha Vantage symbol/market
CRYPTO_TICKER_MAP: Dict[str, Dict[str, str]] = {
    "BTC-USD": {"symbol": "BTC", "market": "USD"},
    # add more here if needed, e.g.:
    # "ETH-USD": {"symbol": "ETH", "market": "USD"},
}


# -------------------------
# Date normalization helper
# -------------------------

def _normalize_date_str(s: Optional[str]) -> Optional[str]:
    """
    Normalize a date-like string (e.g. '2024/01/01', '2024-01-01') to
    'YYYY-MM-DD', which both FMP and Alpha Vantage can work with.
    """
    if s is None:
        return None
    s = str(s).strip()
    if not s:
        return None
    return pd.to_datetime(s).strftime("%Y-%m-%d")


# -------------------------
# Helper: Get API keys
# -------------------------

def _get_fmp_api_key() -> str:
    key = os.getenv("FMP_API_KEY")
    if key:
        return key

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


def _get_alpha_vantage_key() -> str:
    key = os.getenv("ALPHAVANTAGE_API_KEY")
    if key:
        return key

    try:
        import streamlit as st  # type: ignore

        if "ALPHAVANTAGE_API_KEY" in st.secrets:
            return st.secrets["ALPHAVANTAGE_API_KEY"]
    except Exception:
        pass

    raise RuntimeError(
        "ALPHAVANTAGE_API_KEY not found. Set it as an environment variable or "
        "in Streamlit secrets."
    )


# -------------------------
# FMP price + dividend helpers
# -------------------------

def _fetch_fmp_prices_for_ticker(
    ticker: str, start: Optional[str], end: Optional[str]
) -> pd.Series:
    """
    Fetch historical adjusted close prices for a single ticker from FMP.
    Returns a Series indexed by Date with the column name = ticker.
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

    if "adjClose" in df.columns:
        s = df["adjClose"].astype(float)
    else:
        s = df["close"].astype(float)

    s.name = ticker
    s.index.name = "Date"
    return s


def _fetch_fmp_dividends_for_ticker(
    ticker: str, start: Optional[str], end: Optional[str]
) -> pd.Series:
    """
    Fetch historical cash dividends for a single stock/ETF from FMP.
    Returns a Series indexed by Date with dividend amounts.
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
# Alpha Vantage helper for BTC-USD (and other crypto)
# -------------------------

def _fetch_crypto_prices_from_alpha_vantage(
    ticker: str, start: Optional[str], end: Optional[str]
) -> pd.Series:
    """
    Fetch daily close prices in USD for a crypto ticker using Alpha Vantage.

    Uses the DIGITAL_CURRENCY_DAILY endpoint and returns a Series of closes in
    the requested fiat market (e.g. BTC-USD).
    """
    t_upper = ticker.upper()
    if t_upper not in CRYPTO_TICKER_MAP:
        return pd.Series(dtype="float64", name=t_upper)

    symbol = CRYPTO_TICKER_MAP[t_upper]["symbol"]
    market = CRYPTO_TICKER_MAP[t_upper]["market"]
    api_key = _get_alpha_vantage_key()

    url = (
        "https://www.alphavantage.co/query"
        f"?function=DIGITAL_CURRENCY_DAILY&symbol={symbol}"
        f"&market={market}&apikey={api_key}"
    )

    resp = requests.get(url, timeout=20)
    if resp.status_code != 200:
        print(
            f"Warning: Alpha Vantage returned status {resp.status_code} "
            f"for {t_upper}. Skipping this crypto ticker."
        )
        return pd.Series(dtype="float64", name=t_upper)

    data = resp.json()
    ts_key = "Time Series (Digital Currency Daily)"
    if ts_key not in data:
        # Sometimes Alpha Vantage returns an "Error Message" or "Note"
        print(
            f"Warning: Alpha Vantage did not return time series data for {t_upper}. "
            "You may have hit the rate limit (5 calls/min, 500/day) or the API key "
            "is invalid. Skipping this crypto ticker."
        )
        return pd.Series(dtype="float64", name=t_upper)

    ts = data[ts_key]
    df = pd.DataFrame.from_dict(ts, orient="index")
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # Look for the USD close column (e.g. '4b. close (USD)')
    close_col = None
    for col in df.columns:
        if "close" in col.lower() and "(usd)" in col.lower():
            close_col = col
            break
    if close_col is None:
        # Fallback: any column that contains 'close'
        for col in df.columns:
            if "close" in col.lower():
                close_col = col
                break

    if close_col is None:
        print(
            f"Warning: could not find a close column for {t_upper} in "
            "Alpha Vantage data. Skipping."
        )
        return pd.Series(dtype="float64", name=t_upper)

    s = df[close_col].astype(float)
    s.name = t_upper
    s.index.name = "Date"

    # Filter to requested date range
    if start:
        s = s[s.index >= pd.to_datetime(start)]
    if end:
        s = s[s.index <= pd.to_datetime(end)]

    return s


# -------------------------
# Public API used by main.py
# -------------------------

def download_prices(
    tickers: List[str], start: str, end: Optional[str], interval: str = "1d"
) -> pd.DataFrame:
    """
    Download adjusted close prices for a list of tickers.

    - For crypto tickers in CRYPTO_TICKER_MAP (e.g. BTC-USD), use Alpha Vantage.
    - For all others, use FMP.
    - Symbols that cannot be fetched are skipped with a warning.
    """
    norm_start = _normalize_date_str(start)
    norm_end = _normalize_date_str(end)

    series_list = []

    for t in tickers:
        t_upper = t.upper()

        # Crypto via Alpha Vantage
        if t_upper in CRYPTO_TICKER_MAP:
            s = _fetch_crypto_prices_from_alpha_vantage(t_upper, norm_start, norm_end)
            if s.empty:
                print(
                    f"Warning: no Alpha Vantage crypto price data for {t_upper} "
                    "in the requested period."
                )
            else:
                series_list.append(s)
            continue

        # Everything else via FMP
        s = _fetch_fmp_prices_for_ticker(t_upper, norm_start, norm_end)
        if s.empty:
            print(
                f"Warning: no FMP price data for {t_upper} "
                "in the requested period."
            )
        else:
            series_list.append(s)

    if not series_list:
        raise ValueError(
            "No price data returned from FMP/Alpha Vantage for any ticker. "
            "Check your tickers, dates, API keys, or network connectivity."
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
    norm_start = _normalize_date_str(start)
    norm_end = _normalize_date_str(end)

    divs: Dict[str, pd.Series] = {}

    for t in tickers:
        t_upper = t.upper()

        # Crypto: no dividends
        if t_upper in CRYPTO_TICKER_MAP:
            continue

        s = _fetch_fmp_dividends_for_ticker(t_upper, norm_start, norm_end)
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
