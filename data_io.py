import os
from typing import List, Optional, Dict

import requests
import pandas as pd

FMP_BASE_URL = "https://financialmodelingprep.com/api/v3"


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


def _fetch_fmp_prices_for_ticker(
    ticker: str, start: str, end: Optional[str]
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
    resp.raise_for_status()
    data = resp.json()

    hist = data.get("historical", [])
    if not hist:
        # No data for this ticker / date range
        return pd.Series(dtype="float64", name=ticker)

    df = pd.DataFrame(hist)
    # FMP uses "date"
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


def download_prices(
    tickers: List[str], start: str, end: Optional[str], interval: str = "1d"
) -> pd.DataFrame:
    """
    Download adjusted close prices for a list of tickers from FMP.

    interval is kept for compatibility but FMP free tier provides daily
    data only; we ignore non-daily intervals.
    """
    series_list = []
    for t in tickers:
        s = _fetch_fmp_prices_for_ticker(t, start, end)
        if s.empty:
            print(f"Warning: no FMP price data for {t} in the requested period.")
        else:
            series_list.append(s)

    if not series_list:
        raise ValueError(
            "No price data returned from FMP for any ticker. "
            "Check your tickers, dates, or API usage."
        )

    prices = pd.concat(series_list, axis=1)
    prices.index.name = "Date"
    return prices.sort_index()


def _fetch_fmp_dividends_for_ticker(
    ticker: str, start: str, end: Optional[str]
) -> pd.Series:
    """
    Fetch historical cash dividends for a single stock from FMP.

    Returns a Series indexed by Date with dividend amounts (0.0 if none).
    If no dividend data is available, returns an empty Series.
    """
    api_key = _get_fmp_api_key()
    params: Dict[str, str] = {"apikey": api_key}
    if start:
        params["from"] = start
    if end:
        params["to"] = end

    url = f"{FMP_BASE_URL}/historical-price-full/stock_dividend/{ticker}"
    resp = requests.get(url, params=params, timeout=15)

    # Some tickers (ETFs, crypto) simply won't have dividend data
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

    # Prefer adjusted dividend if present
    col = "adjDividend" if "adjDividend" in df.columns else "dividend"
    s = df[col].astype(float)
    s.name = ticker
    s.index.name = "Date"
    return s


def download_dividends(
    tickers: List[str], start: str, end: Optional[str]
) -> pd.DataFrame:
    """
    Download dividend cash flows for all tickers from FMP.

    Returns a DataFrame with Date index and one column per ticker.
    Missing dividends are filled with 0.0.
    """
    divs = {}
    for t in tickers:
        s = _fetch_fmp_dividends_for_ticker(t, start, end)
        if s.empty:
            # No dividends for this ticker in the period
            continue
        divs[t] = s

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


def ensure_output_dir(path: str = "outputs"):
    os.makedirs(path, exist_ok=True)
    return path
