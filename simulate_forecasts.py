import os
import json
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


def _load_current_series(outdir: str) -> Dict[str, pd.Series]:
    """
    Load the realized value series for each portfolio:
    Active, Passive, ORP, Complete.

    Assumes these were already created by main.py:
      - active_portfolio_value.csv
      - passive_portfolio_value.csv
      - clean_prices.csv & summary.json (for ORP)
    And that complete_portfolio was already computed in make_performance_plots.
    """
    # Active & passive
    active = pd.read_csv(
        os.path.join(outdir, "active_portfolio_value.csv"),
        parse_dates=["Date"],
        index_col="Date",
    ).iloc[:, 0]

    passive = pd.read_csv(
        os.path.join(outdir, "passive_portfolio_value.csv"),
        parse_dates=["Date"],
        index_col="Date",
    ).iloc[:, 0]

    # ORP & complete: recompute exactly as in make_performance_plots
    summary_path = os.path.join(outdir, "summary.json")
    prices_path = os.path.join(outdir, "clean_prices.csv")

    if not (os.path.exists(summary_path) and os.path.exists(prices_path)):
        raise FileNotFoundError("Missing summary.json or clean_prices.csv in outputs")

    with open(summary_path, "r") as f:
        summary = json.load(f)

    weights_dict = summary.get("max_sharpe_weights", {})
    if not weights_dict:
        raise ValueError("No max_sharpe_weights in summary.json")

    prices = pd.read_csv(prices_path, parse_dates=["Date"], index_col="Date")
    w = pd.Series(weights_dict)
    w = w[w != 0.0]
    tickers_orp = list(w.index)

    missing = [t for t in tickers_orp if t not in prices.columns]
    if missing:
        raise ValueError(f"Missing prices for {missing} in clean_prices.csv")

    prices_orp = prices[tickers_orp].dropna(how="any")

    # Align on common index with active
    common_index = active.index.intersection(prices_orp.index)
    active = active.loc[common_index]
    passive = passive.loc[common_index]
    prices_orp = prices_orp.loc[common_index]

    # Build ORP series (realized)
    alloc = w * 1_000_000
    p0 = prices_orp.iloc[0]
    shares = alloc / p0
    orp_values = (prices_orp * shares).sum(axis=1)

    # Risk-free leg + complete portfolio
    rf_annual = summary.get("risk_free_rate_annual", 0.04)
    rf_daily = (1 + rf_annual) ** (1 / 252) - 1
    steps = np.arange(len(common_index))
    rf_values = 1_000_000 * (1 + rf_daily) ** steps
    rf_series = pd.Series(rf_values, index=common_index)

    y_cp = 0.8
    if os.path.exists("config.json"):
        with open("config.json", "r") as f:
            cfg = json.load(f)
        y_cp = cfg.get("complete_portfolio", {}).get("y", y_cp)

    complete_values = y_cp * orp_values + (1 - y_cp) * rf_series

    return {
        "Active": active,
        "Passive": passive,
        "ORP": orp_values,
        "Complete": complete_values,
    }


def _estimate_log_stats(series: pd.Series) -> Tuple[float, float]:
    """
    Estimate daily log-return mean and volatility from a value series.
    """
    log_ret = np.log(series / series.shift(1)).dropna()
    mu = log_ret.mean()
    sigma = log_ret.std()
    return float(mu), float(sigma)


def _simulate_paths(
    s0: float,
    mu: float,
    sigma: float,
    horizon_days: int,
    n_paths: int,
) -> np.ndarray:
    """
    Simulate GBM paths for a given starting value.

    dS/S = mu*dt + sigma*dW
    Implemented in log-space.
    """
    dt = 1.0
    # increments: (mu - 0.5 sigma^2) dt + sigma sqrt(dt) Z
    drift = (mu - 0.5 * sigma ** 2) * dt
    shock_scale = sigma * np.sqrt(dt)

    shocks = drift + shock_scale * np.random.randn(horizon_days, n_paths)
    log_paths = np.cumsum(shocks, axis=0)
    paths = s0 * np.exp(log_paths)
    # prepend initial value at t=0
    paths = np.vstack([np.full((1, n_paths), s0), paths])
    return paths


def run_martingale_forecasts(
    outdir: str = "outputs",
    horizon_days: int = 252 * 3,
    n_paths: int = 500,
) -> None:
    """
    Generate forward-looking scenarios for Active, Passive, ORP, and Complete
    using both historical drift and a martingale (zero-drift) assumption.

    Saves:
      - forward_scenarios.png  (fan chart for each portfolio)
    """
    portfolios = _load_current_series(outdir)

    last_date = max(s.index[-1] for s in portfolios.values())
    future_index = pd.date_range(
        start=last_date, periods=horizon_days + 1, freq="B"
    )

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    axes = axes.flatten()

    for ax, (name, series) in zip(axes, portfolios.items()):
        s0 = float(series.iloc[-1])
        mu, sigma = _estimate_log_stats(series)

        # Historical paths
        hist_paths = _simulate_paths(s0, mu, sigma, horizon_days, n_paths)
        hist_q05 = np.percentile(hist_paths, 5, axis=1)
        hist_q50 = np.percentile(hist_paths, 50, axis=1)
        hist_q95 = np.percentile(hist_paths, 95, axis=1)

        # Martingale paths (zero drift in log-returns)
        mart_paths = _simulate_paths(s0, 0.0, sigma, horizon_days, n_paths)
        mart_q05 = np.percentile(mart_paths, 5, axis=1)
        mart_q50 = np.percentile(mart_paths, 50, axis=1)
        mart_q95 = np.percentile(mart_paths, 95, axis=1)

        # Plot realized history
        ax.plot(series.index, series.values, label="Realized", color="black")

        # Historical model fan
        ax.plot(future_index, hist_q50, label="Hist median", color="tab:blue")
        ax.fill_between(
            future_index,
            hist_q05,
            hist_q95,
            color="tab:blue",
            alpha=0.15,
            label="Hist 5–95%",
        )

        # Martingale model fan (dashed)
        ax.plot(
            future_index,
            mart_q50,
            linestyle="--",
            color="tab:orange",
            label="Martingale median",
        )
        ax.fill_between(
            future_index,
            mart_q05,
            mart_q95,
            color="tab:orange",
            alpha=0.15,
            label="Martingale 5–95%",
        )

        ax.set_title(name)
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(
            mtick.FuncFormatter(lambda x, pos: f"${x/1e6:.1f}M")
        )

    fig.suptitle(
        "Forward Scenarios (3 Years): Historical vs Martingale Models",
        fontsize=14,
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4)
    fig.autofmt_xdate()
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = os.path.join(outdir, "forward_scenarios.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved forward scenarios plot to {out_path}")
