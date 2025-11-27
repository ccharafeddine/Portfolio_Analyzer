import os
import pandas as pd
import statsmodels.api as sm

def run_style_regression(outdir: str = "outputs", style_tickers=None):
    if style_tickers is None:
        style_tickers = ["SPYG", "SCHD", "XYLD", "IAU", "BTC-USD"]

    active = pd.read_csv(os.path.join(outdir, "active_portfolio_value.csv"),
                         parse_dates=["Date"], index_col="Date")
    prices = pd.read_csv(os.path.join(outdir, "clean_prices.csv"),
                         parse_dates=["Date"], index_col="Date")

    active_m = active.iloc[:, 0].resample("ME").last().pct_change().dropna()
    styles = prices[style_tickers].resample("ME").last().pct_change().dropna()

    data = pd.concat([active_m, styles], axis=1, join="inner")
    data.columns = ["Rp"] + style_tickers

    Y = data["Rp"]
    X = sm.add_constant(data[style_tickers])

    model = sm.OLS(Y, X).fit()
    print(model.summary())
    return model
