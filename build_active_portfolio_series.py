import os
import pandas as pd
import numpy as np

# ---- Default user settings (can be changed here or moved to config later) ----
CAPITAL = 1_000_000  # total starting capital
START_DATE = "2020-01-02"  # first date you are willing to buy

# Target weights for the ACTIVE portfolio
WEIGHTS = {
    "BTC-USD": 0.08,
    "MARA": 0.02,
    "SPYG": 0.20,
    "XLK": 0.15,
    "AAPL": 0.10,
    "UNH": 0.05,
    "XLP": 0.10,
    "JEPI": 0.10,
    "IAU": 0.10,
    "SHY": 0.10,
}


def build_active_portfolio(
    prices_csv: str = "outputs/clean_prices.csv",
    capital: float = CAPITAL,
    start_date: str = START_DATE,
    weights: dict = WEIGHTS,
    outdir: str = "outputs",
):
    """
    Build the active portfolio series and holdings table.

    Reads prices from prices_csv, finds the first common trading date on/after
    start_date, buys according to 'weights' and 'capital', and then writes:
      - outputs/holdings_table.csv
      - outputs/active_portfolio_value.csv

    Returns (holdings_df, portfolio_value_df).
    """
    os.makedirs(outdir, exist_ok=True)

    # ---- Load prices ----
    prices = pd.read_csv(prices_csv, parse_dates=["Date"], index_col="Date")

    # Restrict to tickers we care about (and that exist in the prices file)
    tickers = [t for t in weights.keys() if t in prices.columns]
    if not tickers:
        raise ValueError("None of the active-portfolio tickers are in clean_prices.csv")

    # Only keep rows on/after the desired start date
    start_date = pd.Timestamp(start_date)
    prices = prices.loc[prices.index >= start_date, tickers]

    # Drop any tickers that have *no* valid prices after the start date
    has_data = prices.notna().any(axis=0)
    good_tickers = list(has_data[has_data].index)

    if not good_tickers:
        raise ValueError("No tickers have any price data after start_date.")

    if len(good_tickers) < len(tickers):
        missing = [t for t in tickers if t not in good_tickers]
        print(
            "Warning: the following tickers have no usable data after start_date "
            f"and will be skipped: {missing}"
        )

    prices = prices[good_tickers]

    # ---- Find purchase date (first date with prices for all remaining tickers) ----
    valid = prices.dropna(how="any")
    if valid.empty:
        raise ValueError(
            "No common date with all remaining tickers after start_date. "
            "Try moving the start date later or removing problematic tickers."
        )

    purchase_date = valid.index[0]
    purchase_prices = valid.loc[purchase_date]


    print(f"Purchase date: {purchase_date.date()}")
    print("Purchase prices:")
    print(purchase_prices)

    # ---- Compute shares per ticker using weights & capital ----
    holdings_rows = []
    shares = {}
    total_invested = 0.0

    for t, w in weights.items():
        if t not in purchase_prices.index:
            print(f"Warning: {t} not in price data on purchase date; skipping.")
            continue

        dollar_allocation = capital * w
        n_shares = np.floor(dollar_allocation / purchase_prices[t])

        invested = n_shares * purchase_prices[t]
        holdings_rows.append(
            {
                "Ticker": t,
                "TargetWeight": w,
                "PurchasePrice": float(purchase_prices[t]),
                "Shares": float(n_shares),
                "Invested": float(invested),
            }
        )
        shares[t] = n_shares
        total_invested += invested

    holdings = pd.DataFrame(holdings_rows)
    if total_invested <= 0:
        raise ValueError("Total invested is zero – check weights, capital, and prices.")

    holdings["RealizedWeight"] = holdings["Invested"] / total_invested

    holdings_path = os.path.join(outdir, "holdings_table.csv")
    holdings.to_csv(holdings_path, index=False)

    # ---- Build portfolio value series ----
    prices_after = prices.loc[prices.index >= purchase_date]
    portfolio_value = pd.DataFrame(index=prices_after.index)
    portfolio_value["PortfolioValue"] = 0.0

    for t, n_shares in shares.items():
        portfolio_value["PortfolioValue"] += prices_after[t] * n_shares

    portfolio_value = portfolio_value.dropna(how="all")

    pv_path = os.path.join(outdir, "active_portfolio_value.csv")
    portfolio_value.to_csv(pv_path)

    print("\nSaved:")
    print(f" - {holdings_path}")
    print(f" - {pv_path}")
    print(
        f"\nStarting portfolio value on {purchase_date.date()}: "
        f"{portfolio_value['PortfolioValue'].iloc[0]:,.2f}"
    )

    return holdings, portfolio_value


if __name__ == "__main__":
    # Allow you to run this file standalone if clean_prices.csv already exists.
    build_active_portfolio()
