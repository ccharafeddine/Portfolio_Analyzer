import os
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.linear_model import RegressionResultsWrapper


DEFAULT_STYLE_TICKERS: Sequence[str] = ["SPYG", "SCHD", "XYLD", "IAU", "BTC-USD"]


def _load_prices(outdir: str) -> pd.DataFrame:
    """
    Load daily prices from clean_prices.csv in the given output directory.
    """
    prices_path = os.path.join(outdir, "clean_prices.csv")
    if not os.path.exists(prices_path):
        raise FileNotFoundError(
            f"clean_prices.csv not found in {outdir}. "
            "Style analysis requires this file."
        )

    prices = pd.read_csv(prices_path, parse_dates=["Date"], index_col="Date")
    # Keep only numeric columns
    num_cols = prices.select_dtypes(include=["number"]).columns
    prices = prices[num_cols]
    return prices


def _compute_monthly_returns_from_series(series: pd.Series) -> pd.Series:
    """
    Convert a daily value/price series into end-of-month simple returns.
    """
    s = series.dropna().astype(float).sort_index()
    if s.empty:
        return pd.Series(dtype=float, name=series.name)

    monthly = s.resample("ME").last()
    rets = monthly.pct_change().dropna()
    rets.name = series.name or "Rp"
    return rets


def _compute_monthly_returns_from_prices(
    prices: pd.DataFrame, tickers: Iterable[str]
) -> pd.DataFrame:
    """
    Compute monthly close-to-close returns for the given tickers from a
    daily price DataFrame.
    """
    cols = [t for t in tickers if t in prices.columns]
    if not cols:
        return pd.DataFrame()

    monthly = prices[cols].dropna(how="all").resample("ME").last()
    rets = monthly.pct_change().dropna(how="all")
    return rets


def _fit_regression(
    active_m: pd.Series,
    style_rets: pd.DataFrame,
) -> Optional[RegressionResultsWrapper]:
    """
    Fit OLS of active monthly returns on style/factor monthly returns.
    """
    if style_rets.empty:
        print("Style regression skipped: style returns DataFrame is empty.")
        return None

    # Align on common dates
    data = pd.concat([active_m, style_rets], axis=1, join="inner").dropna()
    if data.empty:
        print(
            "Style regression skipped: no overlapping monthly data between "
            "active portfolio and style tickers."
        )
        return None

    # Rename columns: dependent = Rp, others = factors
    dep_name = active_m.name or "Rp"
    data.columns = [dep_name] + list(style_rets.columns)

    Y = data[dep_name]
    X = sm.add_constant(data[style_rets.columns])

    model = sm.OLS(Y, X).fit()
    print(model.summary())
    # Attach tickers used as a convenience attribute
    model.style_tickers_ = list(style_rets.columns)
    return model


def _save_regression_summary(
    model: RegressionResultsWrapper, outdir: str
) -> None:
    """
    Save a compact summary of the style regression to CSV for use in the app
    or for inspection.
    """
    if model is None:
        return

    os.makedirs(outdir, exist_ok=True)

    params = model.params
    tvals = model.tvalues
    pvals = model.pvalues

    df = pd.DataFrame(
        {
            "coef": params,
            "t_value": tvals,
            "p_value": pvals,
        }
    )

    # Add a small info row for R^2 etc. (optional)
    info = pd.DataFrame(
        {
            "coef": [np.nan],
            "t_value": [np.nan],
            "p_value": [np.nan],
        },
        index=["_meta_"],
    )
    info.loc["_meta_", "coef"] = model.rsquared
    info.loc["_meta_", "t_value"] = model.rsquared_adj
    info.loc["_meta_", "p_value"] = model.nobs

    df = pd.concat([df, info], axis=0)

    out_path = os.path.join(outdir, "style_regression_summary.csv")
    df.to_csv(out_path)
    print(f"Saved style regression summary to {out_path}")


# ---------------------------------------------------------------------
# Public APIs
# ---------------------------------------------------------------------


def run_style_analysis(
    active_series: pd.Series,
    outdir: str = "outputs",
    style_tickers: Optional[Sequence[str]] = None,
):
    """
    Main entry point used by main.py.

    Parameters
    ----------
    active_series : Series
        Daily active portfolio value series (e.g. from active_portfolio_value.csv).
    outdir : str
        Output directory where clean_prices.csv lives and where results
        will be written.
    style_tickers : sequence of str, optional
        List of style / factor tickers to use. If None, a default set is used.

    Returns
    -------
    model : statsmodels OLS result or None
        The fitted regression model, or None if regression is skipped.
    """
    if style_tickers is None:
        style_tickers = DEFAULT_STYLE_TICKERS

    try:
        prices = _load_prices(outdir)
    except FileNotFoundError as e:
        print(f"Style analysis skipped: {e}")
        return None

    # Monthly active returns from the passed series
    active_m = _compute_monthly_returns_from_series(active_series)

    # Only keep style tickers that are actually present in prices
    available_styles = [t for t in style_tickers if t in prices.columns]

    if not available_styles:
        print(
            "Style regression skipped: none of the requested style tickers "
            "are present in clean_prices.csv."
        )
        return None

    style_rets = _compute_monthly_returns_from_prices(prices, available_styles)

    model = _fit_regression(active_m, style_rets)
    if model is not None:
        _save_regression_summary(model, outdir)

    return model


def run_style_regression(outdir: str = "outputs", style_tickers=None):
    """
    Backwards-compatible wrapper that loads the active portfolio series from
    active_portfolio_value.csv and then delegates to run_style_analysis.

    This preserves the behavior of earlier versions of the project while
    allowing main.py to call run_style_analysis(active_series=..., outdir=...).
    """
    active_path = os.path.join(outdir, "active_portfolio_value.csv")
    if not os.path.exists(active_path):
        print(
            "Style regression skipped: active_portfolio_value.csv not found "
            f"in {outdir}."
        )
        return None

    active_df = pd.read_csv(active_path, parse_dates=["Date"], index_col="Date")
    num_cols = active_df.select_dtypes(include=["number"]).columns
    if len(num_cols) == 0:
        print(
            "Style regression skipped: active_portfolio_value.csv has no "
            "numeric columns."
        )
        return None

    active_series = active_df[num_cols[0]].astype(float).sort_index()
    return run_style_analysis(
        active_series=active_series,
        outdir=outdir,
        style_tickers=style_tickers,
    )
