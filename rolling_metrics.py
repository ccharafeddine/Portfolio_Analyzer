import os
import json
import io
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import imageio.v2 as imageio


def _load_monthly_returns(outdir: str) -> Optional[pd.DataFrame]:
    path = os.path.join(outdir, "monthly_returns.csv")
    if not os.path.exists(path):
        print(f"[rolling_metrics] monthly_returns.csv not found in {outdir}, skipping.")
        return None

    rets = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    # Drop columns that are entirely NaN
    rets = rets.dropna(how="all", axis=1)
    if rets.empty:
        print("[rolling_metrics] monthly_returns.csv is empty after dropping NaNs.")
        return None

    return rets


def _load_config() -> dict:
    cfg_path = "config.json"
    if not os.path.exists(cfg_path):
        return {}
    with open(cfg_path, "r") as f:
        return json.load(f)


def _compute_rolling_stats(
    rets: pd.DataFrame,
    window: int,
    rf_annual: float,
    benchmark: Optional[str],
) -> tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Compute rolling annualized volatility, rolling annualized Sharpe, and
    (optionally) rolling beta vs a benchmark.
    """
    if rets.shape[0] < window:
        raise ValueError(
            f"Not enough observations ({rets.shape[0]}) for rolling window={window}."
        )

    # Rolling annualized volatility
    rolling_vol = rets.rolling(window).std() * np.sqrt(12.0)

    # Rolling annualized mean
    rolling_mean = rets.rolling(window).mean() * 12.0

    # Rolling Sharpe: (E[R] - rf) / vol
    with np.errstate(divide="ignore", invalid="ignore"):
        rolling_sharpe = (rolling_mean - rf_annual) / rolling_vol

    rolling_beta = None
    if benchmark and benchmark in rets.columns:
        bench = rets[benchmark]
        betas = {}
        for col in rets.columns:
            if col == benchmark:
                continue
            cov = rets[col].rolling(window).cov(bench)
            var_m = bench.rolling(window).var()
            with np.errstate(divide="ignore", invalid="ignore"):
                beta = cov / var_m
            betas[col] = beta
        rolling_beta = pd.DataFrame(betas, index=rets.index)

    return rolling_vol, rolling_sharpe, rolling_beta


def _plot_rolling_metrics(
    outdir: str,
    rolling_vol: pd.DataFrame,
    rolling_sharpe: pd.DataFrame,
    rolling_beta: Optional[pd.DataFrame],
) -> None:
    """
    Save rolling_metrics.png with 2–3 stacked panels:
      - Rolling annualized volatility
      - Rolling Sharpe ratio
      - (optional) Rolling beta vs benchmark
    """
    n_panels = 2 + (1 if rolling_beta is not None else 0)
    fig, axes = plt.subplots(
        n_panels,
        1,
        figsize=(12, 3.5 * n_panels),
        sharex=True,
    )

    if n_panels == 1:
        axes = [axes]

    # Panel 1: Rolling volatility
    ax_vol = axes[0]
    for col in rolling_vol.columns:
        ax_vol.plot(rolling_vol.index, rolling_vol[col], label=col, alpha=0.8)
    ax_vol.set_ylabel("Ann. Volatility")
    ax_vol.set_title("Rolling Annualized Volatility")
    ax_vol.grid(True, alpha=0.3)
    ax_vol.yaxis.set_major_formatter(
        mtick.FuncFormatter(lambda x, pos: f"{x:.1%}")
    )

    # Panel 2: Rolling Sharpe
    ax_sharpe = axes[1]
    for col in rolling_sharpe.columns:
        ax_sharpe.plot(rolling_sharpe.index, rolling_sharpe[col], label=col, alpha=0.8)
    ax_sharpe.set_ylabel("Sharpe Ratio")
    ax_sharpe.set_title("Rolling Annualized Sharpe Ratio")
    ax_sharpe.axhline(0.0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax_sharpe.grid(True, alpha=0.3)

    # Panel 3: Rolling beta (if available)
    if rolling_beta is not None:
        ax_beta = axes[2]
        for col in rolling_beta.columns:
            ax_beta.plot(rolling_beta.index, rolling_beta[col], label=col, alpha=0.8)
        ax_beta.set_ylabel("Beta vs Benchmark")
        ax_beta.set_title("Rolling Beta vs Benchmark")
        ax_beta.axhline(1.0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
        ax_beta.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Date")

    # One combined legend at the bottom
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=min(len(labels), 4),
        bbox_to_anchor=(0.5, 0.01),
    )

    fig.tight_layout(rect=[0.02, 0.05, 0.98, 0.95])

    out_path = os.path.join(outdir, "rolling_metrics.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[rolling_metrics] Saved rolling metrics plot to {out_path}")


def _build_corr_frames(
    rets: pd.DataFrame,
    window: int,
) -> list:
    """
    Build a list of image frames (arrays) representing rolling correlation
    heatmaps over time.
    """
    frames: list = []
    cols = rets.columns.tolist()

    for end_idx in range(window - 1, len(rets)):
        window_df = rets.iloc[end_idx - window + 1 : end_idx + 1]
        corr = window_df.corr()

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(corr.values, vmin=-1.0, vmax=1.0, cmap="coolwarm")

        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(cols, rotation=90)
        ax.set_yticks(range(len(cols)))
        ax.set_yticklabels(cols)

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Correlation")

        end_date = window_df.index[-1].date()
        ax.set_title(f"Rolling {window}-Month Correlation (end = {end_date})")

        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150)
        buf.seek(0)
        frames.append(imageio.imread(buf))
        plt.close(fig)

    return frames


def _save_corr_gif(outdir: str, frames: list, duration: float = 0.6) -> None:
    if not frames:
        print("[rolling_metrics] No frames for rolling correlation heatmap GIF.")
        return

    gif_path = os.path.join(outdir, "rolling_corr_heatmap.gif")
    # loop=0  -> infinite looping
    imageio.mimsave(gif_path, frames, duration=duration, loop=0)
    print(f"[rolling_metrics] Saved rolling correlation GIF to {gif_path}")


def run_rolling_metrics(outdir: str = "outputs", window_months: int = 12) -> None:
    """
    Entry point for rolling risk analytics.

    Produces:
      - rolling_metrics.png
      - rolling_corr_heatmap.gif
    """
    rets = _load_monthly_returns(outdir)
    if rets is None:
        return

    cfg = _load_config()
    rf_annual = float(cfg.get("risk_free_rate", 0.02))
    benchmark = cfg.get("benchmark") or cfg.get("passive_benchmark")

    try:
        rolling_vol, rolling_sharpe, rolling_beta = _compute_rolling_stats(
            rets=rets,
            window=window_months,
            rf_annual=rf_annual,
            benchmark=benchmark,
        )
    except ValueError as e:
        print(f"[rolling_metrics] {e}")
        return

    _plot_rolling_metrics(outdir, rolling_vol, rolling_sharpe, rolling_beta)

    # Rolling correlation GIF
    frames = _build_corr_frames(rets, window=window_months)
    _save_corr_gif(outdir, frames)
