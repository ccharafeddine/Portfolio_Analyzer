import argparse
import json
import os

import numpy as np
import pandas as pd

from simulate_forecasts import run_martingale_forecasts

from generate_report import generate_report

from data_io import (
    download_prices,
    download_dividends,
    monthly_returns_from_prices,
    ensure_output_dir,
)
from analytics import (
    summarize_returns,
    max_sharpe,
    efficient_frontier,
    sharpe_ratio,
    capm_regression,
)
from plotting import plot_efficient_frontier, plot_cal, plot_capm_scatter

from build_active_portfolio_series import build_active_portfolio
from build_passive_portfolio_series import build_passive_portfolio
from make_performance_plots import run_performance_plots
from make_additional_plots import run_additional_plots
from style_analysis import run_style_regression


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    outdir = ensure_output_dir()

    tickers = cfg["tickers"]
    benchmark = cfg.get("benchmark", "^GSPC")
    use_tickers = tickers + ([benchmark] if benchmark not in tickers else [])

    # -------- 1) Download data --------
    prices = download_prices(
        use_tickers,
        start=cfg["start"],
        end=cfg["end"],
        interval=cfg["interval"],
    )
    prices.to_csv(os.path.join(outdir, "clean_prices.csv"))

    dividends = download_dividends(
        use_tickers,
        start=cfg["start"],
        end=cfg["end"],
    )
    dividends.to_csv(os.path.join(outdir, "dividends.csv"))

    rets_m = monthly_returns_from_prices(prices).dropna(how="all")
    rets_m.to_csv(os.path.join(outdir, "monthly_returns.csv"))

    # -------- 2) Separate benchmark & stats --------
    bench_col = benchmark
    bench_rets = rets_m[bench_col].dropna()
    asset_rets = rets_m.drop(columns=[bench_col], errors="ignore").dropna(how="all")

    # align assets and benchmark on the same dates
    aligned = asset_rets.join(bench_rets, how="inner")
    asset_rets = aligned[asset_rets.columns]
    bench_rets = aligned[bench_col]

    summary = summarize_returns(asset_rets)
    summary.to_csv(os.path.join(outdir, "summary_stats.csv"))
    cov_m = asset_rets.cov()
    cov_m.to_csv(os.path.join(outdir, "cov_matrix.csv"))

    rf = cfg.get("risk_free_rate", 0.02)  # annual risk-free
    rf_m = (1 + rf) ** (1 / 12) - 1

    # -------- 3) Max Sharpe / ORP / Frontier / CAL --------
    mu_a = (1 + asset_rets.mean()) ** 12 - 1  # annualized means

    res = max_sharpe(
        mu_a.values,
        cov_m.values,
        rf=rf,
        bounds=tuple(cfg["max_allocation_bounds"]),
        short_sales=cfg["short_sales"],
    )

    weights = pd.Series(res.x, index=asset_rets.columns, name="weight").round(6)

    cov_a = cov_m.values * 12.0
    port_mean = float(weights @ mu_a.values)
    port_vol = float(np.sqrt(weights.values @ cov_a @ weights.values))
    port_sharpe = sharpe_ratio(weights.values, mu_a.values, cov_m.values, rf)

    # Efficient frontier and CAL plots
    W, R, V = efficient_frontier(
        mu_a.values,
        cov_m.values,
        cfg["frontier_points"],
        tuple(cfg["max_allocation_bounds"]),
        cfg["short_sales"],
    )
    plot_efficient_frontier(R, V, os.path.join(outdir, "efficient_frontier.png"))
    plot_cal(port_vol, port_mean, rf, os.path.join(outdir, "CAL.png"))

    # -------- 4) CAPM regressions & plots --------
    capm_results = {}
    for t in asset_rets.columns:
        df = pd.concat([asset_rets[t], bench_rets], axis=1, keys=[t, "mkt"]).dropna()
        r = capm_regression(df[t], df["mkt"], rf_m)
        capm_results[t] = r

        ex_i = df[t] - rf_m
        ex_m = df["mkt"] - rf_m
        plot_capm_scatter(
            ex_i,
            ex_m,
            r["alpha"],
            r["beta"],
            os.path.join(outdir, f"capm_{t}.png"),
        )

    # -------- 5) Build Active & Passive portfolios --------
    active_cfg = cfg["active_portfolio"]
    holdings_df, active_series = build_active_portfolio(
        prices_csv=os.path.join(outdir, "clean_prices.csv"),
        capital=active_cfg["capital"],
        start_date=active_cfg["start_date"],
        weights=active_cfg["weights"],
        outdir=outdir,
    )

    print("\n=== Active Portfolio Holdings (Initial Allocation) ===")
    print(holdings_df.to_string(index=False))

    passive_cfg = cfg["passive_portfolio"]
    build_passive_portfolio(
        prices_path=os.path.join(outdir, "clean_prices.csv"),
        benchmark=benchmark,
        capital=passive_cfg["capital"],
        start_date=passive_cfg["start_date"],
        outdir=outdir,
    )

    # -------- 6) Save summary.json BEFORE plotting --------
    summary_payload = {
        "risk_free_rate_annual": rf,
        "max_sharpe_weights": weights.to_dict(),
        "max_sharpe_portfolio": {
            "ann_expected_return": port_mean,
            "ann_volatility": port_vol,
            "sharpe_ratio": port_sharpe,
        },
        "capm": capm_results,
    }

    with open(os.path.join(outdir, "summary.json"), "w") as f:
        json.dump(summary_payload, f, indent=2)

    # -------- 7) Performance plots & extra plots (now ORP exists) --------
    run_performance_plots(outdir=outdir, rf_annual=rf)

    run_additional_plots(
        outdir=outdir,
        orp_weights=weights.to_dict(),
        y_cp=cfg["complete_portfolio"]["y"],
    )

    # -------- 8) Style regression --------
    run_style_regression(outdir=outdir)

    # -------- 9) Forward-looking scenario analysis (historical vs martingale) --------
    run_martingale_forecasts(outdir=outdir, horizon_days=252 * 3, n_paths=500)

    # -------- 10) Generate Markdown report summarizing all results ---------
    generate_report(outdir=outdir, config_path="config.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Portfolio optimization & analysis")
    parser.add_argument(
        "--config",
        type=str,
        default="config.json",
        help="Path to JSON config file (default: config.json)",
    )
    args = parser.parse_args()
    main(args.config)
