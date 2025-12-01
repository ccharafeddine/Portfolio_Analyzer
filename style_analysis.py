import os
import pandas as pd
import statsmodels.api as sm


def run_style_regression(outdir: str = "outputs", style_tickers=None):
    """
    Run a style / factor regression of the active portfolio monthly returns
    against a set of style/factor tickers.

    - If some of the requested style tickers are missing from clean_prices.csv,
      they are silently dropped.
    - If none of the requested style tickers are present, the function returns
      None and prints a message instead of raising an error.
    """

    # Default factors (can be overridden by caller / config)
    if style_tickers is None:
        style_tickers = ["SPYG", "SCHD", "XYLD", "IAU", "BTC-USD"]

    # --- Load data ---
    active_path = os.path.join(outdir, "active_portfolio_value.csv")
    prices_path = os.path.join(outdir, "clean_prices.csv")

    active = pd.read_csv(active_path, parse_dates=["Date"], index_col="Date")
    prices = pd.read_csv(prices_path, parse_dates=["Date"], index_col="Date")

    # Monthly returns for the active portfolio
    active_m = active.iloc[:, 0].resample("ME").last().pct_change().dropna()

    # Only use style/factor tickers that actually exist in the prices file
    available_styles = [t for t in style_tickers if t in prices.columns]

    if not available_styles:
        print(
            "Style regression skipped: none of the requested style tickers "
            f"are present in {prices_path}."
        )
        return None

    # Monthly returns for the available style/factor series
    styles = (
        prices[available_styles]
        .resample("ME")
        .last()
        .pct_change()
        .dropna()
    )

    # Align on common dates
    data = pd.concat([active_m, styles], axis=1, join="inner")

    if data.empty:
        print(
            "Style regression skipped: no overlapping monthly data between "
            "active portfolio and style tickers."
        )
        return None

    data.columns = ["Rp"] + available_styles

    Y = data["Rp"]
    X = sm.add_constant(data[available_styles])

    model = sm.OLS(Y, X).fit()
    print(model.summary())

    # Attach the actual factors used so downstream code can inspect them
    model.style_tickers_ = available_styles
    return model
