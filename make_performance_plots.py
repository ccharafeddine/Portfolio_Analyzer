import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def _normalize_to_million(series: pd.Series) -> pd.Series:
    """Scale any value series so its first value = $1,000,000."""
    s = series.dropna()
    if s.empty:
        return series * np.nan
    return 1_000_000 * series / s.iloc[0]


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# ------------------------------------------------------------
# Main plotting function
# ------------------------------------------------------------
def run_performance_plots(outdir: str = "outputs", rf_annual: float = 0.04) -> None:
    _ensure_dir(outdir)

    # ------------------------------------------------------------
    # Load Active + Passive
    # ------------------------------------------------------------
    active_path = os.path.join(outdir, "active_portfolio_value.csv")
    passive_path = os.path.join(outdir, "passive_portfolio_value.csv")

    active = pd.read_csv(active_path, parse_dates=["Date"], index_col="Date")
    passive = pd.read_csv(passive_path, parse_dates=["Date"], index_col="Date")

    active = active.rename(columns={active.columns[0]: "Active"})
    passive = passive.rename(columns={passive.columns[0]: "Passive"})

    df_ap = active.join(passive, how="inner")

    # ------------------------------------------------------------
    # Load clean prices (daily)
    # ------------------------------------------------------------
    price_path = os.path.join(outdir, "clean_prices.csv")
    prices = pd.read_csv(price_path, parse_dates=["Date"], index_col="Date")

    # Forward-fill prices to avoid NaN gaps for newer ETFs
    prices = prices.ffill()

    # ------------------------------------------------------------
    # Load ORP weights
    # ------------------------------------------------------------
    summary_path = os.path.join(outdir, "summary.json")
    if not os.path.exists(summary_path):
        print("[WARN] No summary.json → skipping ORP.")
        return

    with open(summary_path, "r") as f:
        summary = json.load(f)

    if "max_sharpe_weights" not in summary:
        print("[WARN] No max_sharpe_weights in summary → skipping ORP.")
        return

    w = pd.Series(summary["max_sharpe_weights"], dtype=float)
    w = w[w != 0.0]

    # Restrict to tickers we actually have prices for
    available = [t for t in w.index if t in prices.columns]
    if not available:
        print("[WARN] None of ORP tickers found in clean_prices → skipping ORP.")
        return

    w = w.loc[available]

    # Daily price frame for ORP tickers
    prices_orp = prices[available].dropna(how="all")

    # ------------------------------------------------------------
    # Align active/passive and ORP to common dates
    # ------------------------------------------------------------
    common_index = df_ap.index.intersection(prices_orp.index)
    if common_index.empty:
        print("[WARN] Active/Passive and ORP do not overlap in dates.")
        return

    df_ap = df_ap.loc[common_index]
    prices_orp = prices_orp.loc[common_index]

    # ------------------------------------------------------------
    # Realized ORP from daily prices
    # ------------------------------------------------------------
    alloc = w * 1_000_000
    initial_prices = prices_orp.iloc[0]
    shares = alloc / initial_prices

    orp_values = (prices_orp * shares).sum(axis=1)

    # ------------------------------------------------------------
    # Expected ORP (monthly smoothed)
    # ------------------------------------------------------------
    monthly_orp = prices_orp.resample("ME").last().pct_change().dropna()
    r_orp_monthly = monthly_orp[available].dot(w)

    expected_orp_monthly = 1_000_000 * (1 + r_orp_monthly).cumprod()

    expected_orp_daily = pd.Series(index=common_index, dtype=float)
    expected_orp_daily.iloc[0] = 1_000_000
    for dt, val in expected_orp_monthly.items():
        if dt in expected_orp_daily.index:
            expected_orp_daily.loc[dt] = val
    expected_orp_daily = expected_orp_daily.ffill()

    # ------------------------------------------------------------
    # Risk-free daily series
    # ------------------------------------------------------------
    rf_daily = (1 + rf_annual) ** (1 / 252) - 1
    rf_series = 1_000_000 * (1 + rf_daily) ** np.arange(len(common_index))
    rf_series = pd.Series(rf_series, index=common_index)

    # ------------------------------------------------------------
    # Complete portfolio (daily)
    # ------------------------------------------------------------
    y_cp = summary.get("y_cp", 1.0)
    complete_values = y_cp * orp_values + (1 - y_cp) * rf_series

    # ------------------------------------------------------------
    # Save ORP and Complete CSVs
    # ------------------------------------------------------------
    pd.DataFrame({"ORP_Value": orp_values}).to_csv(
        os.path.join(outdir, "orp_value_realized.csv"),
        index_label="Date",
    )

    pd.DataFrame({"Complete_Value": complete_values}).to_csv(
        os.path.join(outdir, "complete_portfolio_value.csv"),
        index_label="Date",
    )

    # ------------------------------------------------------------
    # Plot: ORP Real vs Expected
    # ------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    plt.plot(orp_values.index, orp_values, label="Realized ORP (daily)")
    plt.plot(expected_orp_daily.index, expected_orp_daily, linestyle="--",
             label="Expected ORP (monthly-smoothed)")
    ax = plt.gca()
    ax.yaxis.set_major_formatter(
        mtick.FuncFormatter(lambda x, pos: f"${x/1e6:.1f}M")
    )
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value (Millions)")
    plt.title("Optimal Risky Portfolio: Realized vs Expected")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "orp_real_vs_expected.png"), dpi=300)
    plt.close()

    # ------------------------------------------------------------
    # Plot: Growth of all portfolios
    # ------------------------------------------------------------
    combined = pd.DataFrame({
        "Active": df_ap["Active"],
        "Passive": df_ap["Passive"],
        "ORP": orp_values,
        "Complete": complete_values,
    })

    growth = combined.apply(_normalize_to_million)

    plt.figure(figsize=(14, 8))
    plt.plot(growth.index, growth["Active"], label="Active")
    plt.plot(growth.index, growth["Passive"], label="Passive")
    plt.plot(growth.index, growth["ORP"], label="ORP")
    plt.plot(growth.index, growth["Complete"], label="Complete")

    ax = plt.gca()
    ax.yaxis.set_major_formatter(
        mtick.FuncFormatter(lambda x, pos: f"${x/1e6:.1f}M")
    )

    plt.xlabel("Date")
    plt.ylabel("Portfolio Value (Millions)")
    plt.title("Growth of $1,000,000: Active vs Passive vs ORP vs Complete")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "growth_all_portfolios.png"), dpi=300)
    plt.close()

    print("Saved performance plots in", outdir)
