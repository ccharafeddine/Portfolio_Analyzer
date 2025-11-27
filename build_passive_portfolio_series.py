import os
import pandas as pd

def build_passive_portfolio(
    prices_path: str,
    benchmark: str,
    capital: float,
    start_date: str,
    outdir: str = "outputs"
):
    """
    Build passive buy-and-hold benchmark value series.

    - prices_path: path to clean_prices.csv
    - benchmark: ticker for index, e.g. '^GSPC'
    - capital: starting capital
    - start_date: 'YYYY-MM-DD'
    """
    os.makedirs(outdir, exist_ok=True)

    prices = pd.read_csv(prices_path, parse_dates=["Date"], index_col="Date")

    if benchmark not in prices.columns:
        raise ValueError(f"Benchmark {benchmark} not found in prices columns.")

    start_ts = pd.Timestamp(start_date)

    bench_series = prices[benchmark]
    bench_valid = bench_series[bench_series.index >= start_ts].dropna()
    if bench_valid.empty:
        raise ValueError(f"No valid {benchmark} prices found on or after {start_ts.date()}")

    purchase_date = bench_valid.index[0]
    purchase_price = float(bench_valid.iloc[0])
    units = capital / purchase_price

    bench_after = bench_series[bench_series.index >= purchase_date].copy()
    passive_value = (bench_after * units).to_frame(name="Passive")

    out_path = os.path.join(outdir, "passive_portfolio_value.csv")
    passive_value.to_csv(out_path)

    print(f"[PASSIVE] Benchmark: {benchmark}")
    print(f"[PASSIVE] Purchase date: {purchase_date.date()}, price={purchase_price:.4f}")
    print(f"[PASSIVE] Units: {units:.4f}")
    print(f"[PASSIVE] Saved value series to {out_path}")

    return passive_value
