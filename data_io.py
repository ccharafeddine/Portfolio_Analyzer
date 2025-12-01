
import os
import pandas as pd
import yfinance as yf
from typing import List, Optional

def download_prices(tickers: List[str], start: str, end: Optional[str], interval: str = "1d") -> pd.DataFrame:
    """Download Adj Close for tickers from Yahoo Finance."""
    try:
        df = yf.download(
            tickers,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=False,
            progress=False,
        )["Adj Close"]
    except Exception as e:
        # Bubble up a clean error that Streamlit can show
        raise ValueError(
            f"Error downloading price data from Yahoo Finance: {e}. "
            "This often means Yahoo is rate limiting. Please try again in a few minutes."
        )

    if isinstance(df, pd.Series):
        df = df.to_frame()

    if df.empty:
        raise ValueError(
            "No price data returned from Yahoo Finance. "
            "This can happen if tickers are invalid, dates are out of range, "
            "or Yahoo is rate limiting. Please try again later or adjust the dates."
        )

    return df.sort_index()

def download_dividends(tickers: List[str], start: str, end: Optional[str]) -> pd.DataFrame:
    """Download dividends and align by date (sum per day)."""
    divs = {}
    for t in tickers:
        div = yf.Ticker(t).dividends
        if len(div) == 0:
            continue
        if start:
            div = div[(div.index >= start)]
        if end:
            div = div[(div.index <= end)]
        divs[t] = div
    if not divs:
        return pd.DataFrame()
    div_df = pd.DataFrame(divs).fillna(0.0)
    return div_df.sort_index()

def monthly_returns_from_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute monthly close-to-close returns from daily Adj Close."""
    monthly = prices.resample("ME").last()
    rets = monthly.pct_change().dropna(how="all")
    return rets

def ensure_output_dir(path: str = "outputs"):
    os.makedirs(path, exist_ok=True)
    return path
