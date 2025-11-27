import os
import json
from math import sqrt

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd


def _normalize_to_million(series: pd.Series) -> pd.Series:
    """Scale any value series so that the first point equals $1,000,000."""
    series = series.dropna()
    if series.empty:
        return series
    return 1_000_000 * series / series.iloc[0]


def run_performance_plots(outdir: str = "outputs", rf_annual: float = 0.04) -> None:
    """
    Create performance charts for active, passive, ORP, and complete portfolios.

    Expects the following files in `outdir` (created earlier by main.py):

    - active_portfolio_value.csv: Date, Value
    - passive_portfolio_value.csv: Date, Value
    - clean_prices.csv: daily adjusted close prices for all tickers
    - summary.json: contains max_sharpe_weights and risk_free_rate_annual

    And `config.json` in the project root for complete_portfolio.y.
    """

    # ------------------------------------------------------------
    # Load active & passive portfolio value series
    # ------------------------------------------------------------
    active_path = os.path.join(outdir, "active_portfolio_value.csv")
    passive_path = os.path.join(outdir, "passive_portfolio_value.csv")

    active = pd.read_csv(active_path, parse_dates=["Date"], index_col="Date")
    passive = pd.read_csv(passive_path, parse_dates=["Date"], index_col="Date")

    active = active.rename(columns={active.columns[0]: "Active"})
    passive = passive.rename(columns={passive.columns[0]: "Passive"})

    # Align on common dates first
    df_ap = active.join(passive, how="inner")

    # ------------------------------------------------------------
    # Risk / Return stats for Active vs Passive (annualized)
    # ------------------------------------------------------------
    monthly = df_ap.resample("ME").last().pct_change().dropna()

    rf = rf_annual
    rf_m = (1 + rf) ** (1 / 12) - 1

    ret_m_active = monthly["Active"].mean()
    ret_m_passive = monthly["Passive"].mean()
    vol_m_active = monthly["Active"].std()
    vol_m_passive = monthly["Passive"].std()

    ret_a_active = (1 + ret_m_active) ** 12 - 1
    ret_a_passive = (1 + ret_m_passive) ** 12 - 1
    vol_a_active = vol_m_active * sqrt(12)
    vol_a_passive = vol_m_passive * sqrt(12)

    ex_a = ret_a_active - rf
    ex_p = ret_a_passive - rf
    sharpe_active = ex_a / vol_a_active if vol_a_active != 0 else float("nan")
    sharpe_passive = ex_p / vol_a_passive if vol_a_passive != 0 else float("nan")

    print("\n=== Risk/Return Summary (Annualized) ===")
    print(
        f"Active  - Return: {ret_a_active:.2%}, Vol: {vol_a_active:.2%}, Sharpe: {sharpe_active:.2f}"
    )
    print(
        f"Passive - Return: {ret_a_passive:.2%}, Vol: {vol_a_passive:.2%}, Sharpe: {sharpe_passive:.2f}"
    )

    # ------------------------------------------------------------
    # Plot 1: Growth of $1M, Active vs Passive only
    # ------------------------------------------------------------
    growth_ap = pd.DataFrame(index=df_ap.index)
    growth_ap["Active"] = _normalize_to_million(df_ap["Active"])
    growth_ap["Passive"] = _normalize_to_million(df_ap["Passive"])

    plt.figure(figsize=(10, 6))
    plt.plot(growth_ap.index, growth_ap["Active"], label="Active Portfolio")
    plt.plot(growth_ap.index, growth_ap["Passive"], label="Passive (S&P 500)")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value ($)")
    plt.title("Growth of $1,000,000: Active vs Passive")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "active_vs_passive_growth.png"), dpi=300)
    plt.close()

    # ------------------------------------------------------------
    # Plot 2: Risk–Return scatter (Active vs Passive)
    # ------------------------------------------------------------
    plt.figure(figsize=(7, 6))
    plt.scatter(vol_a_active, ret_a_active, label="Active", marker="o")
    plt.scatter(vol_a_passive, ret_a_passive, label="Passive", marker="s")

    for x, y, label in [
        (vol_a_active, ret_a_active, "Active"),
        (vol_a_passive, ret_a_passive, "Passive"),
    ]:
        plt.annotate(label, (x, y), textcoords="offset points", xytext=(5, 5))

    plt.xlabel("Annualized Volatility")
    plt.ylabel("Annualized Return")
    plt.title("Risk–Return Comparison: Active vs Passive")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "active_vs_passive_risk_return.png"), dpi=300)
    plt.close()

    # ------------------------------------------------------------
    # Plot 3: Last 12 months of monthly returns
    # ------------------------------------------------------------
    last_12 = monthly.tail(12)
    x = np.arange(len(last_12))

    plt.figure(figsize=(12, 6))
    plt.bar(x - 0.2, last_12["Active"], width=0.4, label="Active")
    plt.bar(x + 0.2, last_12["Passive"], width=0.4, label="Passive")
    plt.xticks(
        x, [d.strftime("%Y-%m") for d in last_12.index], rotation=45, ha="right"
    )
    plt.ylabel("Monthly Return")
    plt.title("Last 12 Months of Monthly Returns")
    plt.grid(True, axis="y", linestyle="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(outdir, "active_vs_passive_monthly_returns.png"), dpi=300
    )
    plt.close()

    # ------------------------------------------------------------
    # ORP & Complete Portfolio series from daily prices + weights
    # ------------------------------------------------------------
    summary_path = os.path.join(outdir, "summary.json")
    if not os.path.exists(summary_path):
        print("summary.json not found; skipping ORP/Complete in performance plots.")
        print("Saved performance plots in", outdir)
        return

    with open(summary_path, "r") as f:
        summary = json.load(f)

    weights_dict = summary.get("max_sharpe_weights", {})
    if not weights_dict:
        print("No max_sharpe_weights in summary.json; skipping ORP/Complete.")
        print("Saved performance plots in", outdir)
        return

    prices_path = os.path.join(outdir, "clean_prices.csv")
    prices = pd.read_csv(prices_path, parse_dates=["Date"], index_col="Date")

    w = pd.Series(weights_dict)
    w = w[w != 0.0]  # drop zero-weight assets
    tickers_orp = list(w.index)

    missing = [t for t in tickers_orp if t not in prices.columns]
    if missing:
        print(f"Warning: Missing prices for {missing}; skipping ORP/Complete.")
        print("Saved performance plots in", outdir)
        return

    prices_orp = prices[tickers_orp].dropna(how="any")

    # Align all series on common date range
    common_index = df_ap.index.intersection(prices_orp.index)
    if common_index.empty:
        print(
            "No common dates between active/passive and ORP prices; skipping ORP/Complete."
        )
        print("Saved performance plots in", outdir)
        return

    df_ap = df_ap.loc[common_index]
    prices_orp = prices_orp.loc[common_index]

    # --- Realized ORP: build from daily prices and weights ---
    alloc = w * 1_000_000
    p0 = prices_orp.iloc[0]
    shares = alloc / p0
    orp_values = (prices_orp * shares).sum(axis=1)

    # --- Expected / smoothed ORP: monthly returns + forward-fill to daily ---
    monthly_orp = prices_orp.resample("ME").last().pct_change().dropna()
    r_orp_month = monthly_orp[tickers_orp].dot(w)

    expected_orp_monthly = 1_000_000 * (1 + r_orp_month).cumprod()

    # Build daily series starting at 1,000,000 and stepping at month ends
    expected_orp_daily = pd.Series(index=common_index, dtype=float)
    expected_orp_daily.iloc[0] = 1_000_000

    # assign monthly values on their dates (if they fall inside our daily index)
    for dt, val in expected_orp_monthly.items():
        if dt in expected_orp_daily.index:
            expected_orp_daily.loc[dt] = val

    expected_orp_daily = expected_orp_daily.ffill()

    # --- Risk-free leg for complete portfolio (same as before) ---
    rf_daily = (1 + rf) ** (1 / 252) - 1
    steps = np.arange(len(common_index))
    rf_values = 1_000_000 * (1 + rf_daily) ** steps
    rf_series = pd.Series(rf_values, index=common_index)

    # Load y from config.json (project root)
    y_cp = 0.8
    if os.path.exists("config.json"):
        with open("config.json", "r") as f:
            cfg = json.load(f)
        y_cp = cfg.get("complete_portfolio", {}).get("y", y_cp)

    complete_values = y_cp * orp_values + (1 - y_cp) * rf_series

    # --- Save ORP & Complete realized value series for reporting ---
    orp_df = pd.DataFrame({"ORP_Value": orp_values})
    orp_df.index.name = "Date"
    orp_df.to_csv(os.path.join(outdir, "orp_value_realized.csv"))

    complete_df = pd.DataFrame({"Complete_Value": complete_values})
    complete_df.index.name = "Date"
    complete_df.to_csv(os.path.join(outdir, "complete_portfolio_value.csv"))

    # Optional: save realized ORP weights as well (used in PDF appendix)
    orp_weights_path = os.path.join(outdir, "orp_weights_realized.csv")
    w.to_csv(orp_weights_path, header=["Weight"])

    # ------------------------------------------------------------
    # NEW Plot: Real vs Expected ORP
    # ------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    plt.plot(
        orp_values.index,
        orp_values,
        label="Realized ORP (daily prices)",
    )
    plt.plot(
        expected_orp_daily.index,
        expected_orp_daily,
        linestyle="--",
        label="Expected ORP (monthly-smoothed)",
    )

    ax = plt.gca()
    ax.yaxis.set_major_formatter(
        mtick.FuncFormatter(lambda x, pos: f"${x/1e6:.1f}M")
    )

    plt.xlabel("Date")
    plt.ylabel("Portfolio Value (Millions of $)")
    plt.title("Optimal Risky Portfolio: Realized vs Expected")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "orp_real_vs_expected.png"), dpi=300)
    plt.close()

    # ------------------------------------------------------------
    # Plot 4: Growth of $1M including ORP & Complete (unchanged semantics)
    # ------------------------------------------------------------
    combined = pd.DataFrame(
        {
            "Active": df_ap["Active"],
            "Passive": df_ap["Passive"],
            "ORP": orp_values,
            "Complete": complete_values,
        },
        index=common_index,
    )

    growth_all = combined.apply(_normalize_to_million)

    plt.figure(figsize=(14, 8))
    plt.plot(growth_all.index, growth_all["Active"], label="Active Portfolio")
    plt.plot(growth_all.index, growth_all["Passive"], label="Passive Portfolio")
    plt.plot(growth_all.index, growth_all["ORP"], label="ORP")
    plt.plot(growth_all.index, growth_all["Complete"], label="Complete Portfolio")

    ax = plt.gca()
    ax.yaxis.set_major_formatter(
        mtick.FuncFormatter(lambda x, pos: f"${x/1e6:.1f}M")
    )

    plt.xlabel("Date")
    plt.ylabel("Portfolio Value (Millions of $)")
    plt.title("Growth of $1,000,000: Active vs Passive vs ORP vs Complete")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "growth_all_portfolios.png"), dpi=300)
    plt.close()

    print("Saved performance plots in", outdir)
