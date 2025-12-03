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
    compute_all_portfolio_drawdown_and_tail_risk,
)

from plotting import plot_efficient_frontier, plot_cal, plot_capm_scatter

from build_active_portfolio_series import build_active_portfolio
from build_passive_portfolio_series import build_passive_portfolio

from make_performance_plots import run_performance_plots, run_drawdown_tail_plots
from make_additional_plots import run_additional_plots
from style_analysis import run_style_regression
from rolling_metrics import run_rolling_metrics

from factor_loader import load_factors
from multi_factor_regression import run_all_factor_models


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    outdir = ensure_output_dir()

    # ------------------------------
    # 1) Tickers / dates / benchmark
    # ------------------------------
    tickers = cfg["tickers"]
    benchmark = cfg.get("benchmark", cfg.get("passive_benchmark", "^GSPC"))

    start = cfg.get("start") or cfg.get("start_date")
    end = cfg.get("end") or cfg.get("end_date")
    interval = cfg.get("interval", "1d")

    if start is None or end is None:
        raise ValueError("Config must specify 'start' and 'end' dates.")

    rf = cfg.get("risk_free_rate", 0.02)  # annual risk-free
    short_sales_flag = bool(cfg.get("short_sales", False))
    frontier_points = int(cfg.get("frontier_points", 50))
    y_cp = cfg.get("complete_portfolio", {}).get("y", cfg.get("y_cp", 1.0))

    # ------------------------------
    # 2) Price & dividend data
    # ------------------------------
    use_tickers = sorted(set(tickers + [benchmark]))

    prices = download_prices(
        use_tickers,
        start=start,
        end=end,
        interval=interval,
    )
    prices.to_csv(os.path.join(outdir, "clean_prices.csv"))

    dividends = download_dividends(
        use_tickers,
        start=start,
        end=end,
    )
    dividends.to_csv(os.path.join(outdir, "dividends.csv"))

    # monthly returns
    rets_m = monthly_returns_from_prices(prices)
    rets_m.to_csv(os.path.join(outdir, "monthly_returns.csv"))

    bench_col = benchmark
    if bench_col not in rets_m.columns:
        raise ValueError(f"Benchmark {benchmark} is not in monthly returns columns.")

    bench_rets = rets_m[bench_col].dropna()
    asset_rets = rets_m.drop(columns=[bench_col], errors="ignore").dropna(how="all")

    # align on common dates
    aligned = asset_rets.join(bench_rets, how="inner")
    asset_rets = aligned[asset_rets.columns]
    bench_rets = aligned[bench_col]

    # summary stats & covariance
    summary = summarize_returns(asset_rets)
    summary.to_csv(os.path.join(outdir, "summary_stats.csv"))

    cov_m = asset_rets.cov()
    cov_m.to_csv(os.path.join(outdir, "cov_matrix.csv"))

    # ------------------------------
    # 3) ORP & efficient frontier
    # ------------------------------
    mu_a = (1.0 + asset_rets.mean()) ** 12 - 1.0  # annualized means

    res = max_sharpe(
        mu_a.values,
        cov_m.values,
        rf=rf,
        bounds=tuple(cfg["max_allocation_bounds"]),
        short_sales=short_sales_flag,
    )

    weights = pd.Series(res.x, index=asset_rets.columns, name="weight").round(6)

    cov_a = cov_m.values * 12.0
    port_mean = float(weights @ mu_a.values)
    port_vol = float(np.sqrt(weights.values @ cov_a @ weights.values))
    port_sharpe = sharpe_ratio(weights.values, mu_a.values, cov_m.values, rf)

    # efficient frontier and CAL plots
    W, R, V = efficient_frontier(
        mu_a.values,
        cov_m.values,
        frontier_points,
        tuple(cfg["max_allocation_bounds"]),
        short_sales_flag,
    )
    plot_efficient_frontier(R, V, os.path.join(outdir, "efficient_frontier.png"))
    # call plot_cal with positional args to match your plotting.py signature
    plot_cal(
        port_vol,
        port_mean,
        rf,
        os.path.join(outdir, "CAL.png"),
    )

    # ------------------------------
    # 4) CAPM regressions & plots
    # ------------------------------
    rf_m = (1 + rf) ** (1 / 12) - 1  # monthly rf approximation
    capm_results: dict[str, dict] = {}
    capm_rows: list[dict] = []

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

        row = {"Asset": t}
        row.update(r)
        capm_rows.append(row)

    # save CAPM summary table for the app
    if capm_rows:
        capm_df = pd.DataFrame(capm_rows)
        capm_df.to_csv(os.path.join(outdir, "capm_results.csv"), index=False)

    # ------------------------------
    # 5) Build active & passive portfolios
    # ------------------------------
    active_cfg = cfg["active_portfolio"]
    build_active_portfolio(
        prices_csv=os.path.join(outdir, "clean_prices.csv"),
        capital=active_cfg["capital"],
        start_date=active_cfg["start_date"],
        weights=active_cfg["weights"],
        outdir=outdir,
    )

    passive_cfg = cfg["passive_portfolio"]
    build_passive_portfolio(
        prices_path=os.path.join(outdir, "clean_prices.csv"),
        benchmark=benchmark,
        capital=passive_cfg["capital"],
        start_date=passive_cfg["start_date"],
        outdir=outdir,
    )

    # ------------------------------
    # 6) Style regression (moved AFTER active_portfolio_value.csv exists)
    # ------------------------------
    run_style_regression(outdir=outdir)

    # ------------------------------
    # 7) Save summary.json BEFORE performance plots
    # ------------------------------
    summary_payload = {
        "risk_free_rate_annual": rf,
        "max_sharpe_weights": weights.to_dict(),
        "max_sharpe_portfolio": {
            "ann_expected_return": port_mean,
            "ann_volatility": port_vol,
            "sharpe_ratio": port_sharpe,
        },
        "capm": capm_results,
        "y_cp": y_cp,
    }

    with open(os.path.join(outdir, "summary.json"), "w") as f:
        json.dump(summary_payload, f, indent=2)

    # ------------------------------
    # 8) Performance plots & additional plots
    # ------------------------------
    run_performance_plots(outdir=outdir, rf_annual=rf)

    run_additional_plots(
        outdir=outdir,
        orp_weights=weights.to_dict(),
        y_cp=y_cp,
    )

    # ------------------------------
    # 9) Drawdown & tail-risk analytics
    # ------------------------------
    try:
        compute_all_portfolio_drawdown_and_tail_risk(outdir=outdir, rf_annual=rf)
        run_drawdown_tail_plots(
            outdir=outdir,
            var_level=0.95,
            focus_portfolio="Active",
        )
    except Exception as e:
        print(f"[drawdown/tail] Warning: drawdown & tail-risk analytics failed: {e}")

    # ------------------------------
    # 10) Multi-factor regressions (FF3, Carhart4, FF5, Quality/LowVol)
    # ------------------------------
    factors_dict: dict[str, pd.DataFrame] = {}
    for model in ["ff3", "carhart4", "ff5", "quality_lowvol"]:
        try:
            fdf = load_factors(model, start=start, end=end)
            if fdf is not None and not fdf.empty:
                factors_dict[model] = fdf
        except Exception as e:
            print(f"[factor_loader] Failed to load {model}: {e}")

    if factors_dict:
        run_all_factor_models(
            asset_rets=asset_rets,
            factors_dict=factors_dict,
            outdir=outdir,
        )
    else:
        print("[factor regression] No factor data loaded; skipping multi-factor regressions.")

    # ------------------------------
    # 11) Forward-looking forecasts
    # ------------------------------
    run_martingale_forecasts(outdir=outdir, horizon_days=252 * 3, n_paths=500)

    # ------------------------------
    # 12) Rolling risk analytics
    # ------------------------------
    try:
        run_rolling_metrics(outdir=outdir)
    except Exception as e:
        print(f"[rolling_metrics] Warning: rolling risk analytics failed: {e}")

    # ------------------------------
    # 13) Final markdown/PDF report
    # ------------------------------
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
