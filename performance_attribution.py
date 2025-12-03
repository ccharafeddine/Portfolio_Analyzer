# ------------------------------------------------------------
# performance_attribution.py
# Brinson–Fachler Performance Attribution
# ------------------------------------------------------------

import os
import json
from typing import Dict, Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------

def _safe_read_csv(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        print(f"[attribution] Missing file: {path}")
        return None
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"[attribution] Could not read {path}: {e}")
        return None


def _load_config(config_path: str = "config.json") -> Optional[Dict]:
    if not os.path.exists(config_path):
        print("[attribution] Config not found.")
        return None
    try:
        with open(config_path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"[attribution] Config read error: {e}")
        return None


# ------------------------------------------------------------
# Core Brinson–Fachler formulas
# ------------------------------------------------------------

def compute_brinson_fachler(wP: pd.Series,
                            wB: pd.Series,
                            rP_i: pd.Series,
                            rB_i: pd.Series) -> pd.DataFrame:
    """
    All inputs: pandas Series aligned on same index of buckets
    (e.g., tickers or sectors).
    Returns DataFrame with Allocation, Selection, Interaction, Total.
    """
    # Ensure alignment
    idx = (
        wP.index
        .intersection(wB.index)
        .intersection(rP_i.index)
        .intersection(rB_i.index)
    )
    wP = wP.loc[idx]
    wB = wB.loc[idx]
    rP_i = rP_i.loc[idx]
    rB_i = rB_i.loc[idx]

    if wP.empty:
        raise ValueError("No overlapping buckets available for attribution.")

    R_B = float((wB * rB_i).sum())

    alloc = (wP - wB) * (rB_i - R_B)
    select = wB * (rP_i - rB_i)
    inter = (wP - wB) * (rP_i - rB_i)
    total = alloc + select + inter

    df = pd.DataFrame(
        {
            "wP": wP,
            "wB": wB,
            "rP": rP_i,
            "rB": rB_i,
            "Allocation": alloc,
            "Selection": select,
            "Interaction": inter,
            "Total": total,
        }
    )
    return df


# ------------------------------------------------------------
# Plotting
# ------------------------------------------------------------

def _plot_attribution(df: pd.DataFrame,
                      outpath: str,
                      title: str = "Brinson–Fachler Performance Attribution") -> None:
    """Simple stacked bar chart of Allocation / Selection / Interaction."""
    try:
        buckets = df.index.astype(str)
        A = df["Allocation"]
        S = df["Selection"]
        I = df["Interaction"]

        plt.figure(figsize=(12, 6))
        plt.bar(buckets, A, label="Allocation")
        plt.bar(buckets, S, bottom=A, label="Selection")
        plt.bar(buckets, I, bottom=A + S, label="Interaction")

        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Contribution")
        plt.title(title)
        plt.legend()
        plt.tight_layout()

        plt.savefig(outpath, dpi=150)
        plt.close()

        print(f"[attribution] Saved: {outpath}")
    except Exception as e:
        print(f"[attribution] Plotting failed: {e}")


# ------------------------------------------------------------
# Sector helpers
# ------------------------------------------------------------

def _auto_build_sector_map(tickers: List[str]) -> pd.DataFrame:
    """
    Try to infer sectors for the given tickers using yfinance.
    This is best-effort and will fall back to 'Unknown' on failure.

    Returns a DataFrame with columns ['Ticker', 'Sector'].
    """
    records = []
    for t in tickers:
        sector = None
        t_clean = t.strip().upper()

        # Simple crypto heuristic
        if t_clean.endswith("-USD"):
            sector = "Crypto"

        if sector is None:
            try:
                info = yf.Ticker(t_clean).info
                sector = info.get("sector") or info.get("industry")
            except Exception as e:
                print(
                    f"[attribution] Could not fetch sector for {t_clean} via "
                    f"yfinance: {e}"
                )
                sector = None

        if sector is None or str(sector).strip() == "":
            sector = "Unknown"

        records.append({"Ticker": t_clean, "Sector": sector})

    df = pd.DataFrame.from_records(records)
    print(
        "[attribution] Built sector map from yfinance for tickers: "
        + ", ".join(df["Ticker"])
    )
    return df


def _load_or_build_sector_map(config: Dict,
                              outdir: str,
                              tickers: List[str]) -> Optional[pd.DataFrame]:
    """
    Try to load a sector map CSV with columns: Ticker, Sector.

    Search order:
      1) config["sector_map_path"] (if present)
      2) <outdir>/sector_map.csv
      3) ./sector_map.csv

    If none found, attempt to build one online via yfinance, save it to
    <outdir>/sector_map.csv, and return it.

    Returns a DataFrame or None if everything fails.
    """
    candidates: List[str] = []

    cfg_path = config.get("sector_map_path")
    if cfg_path:
        candidates.append(cfg_path)

    candidates.append(os.path.join(outdir, "sector_map.csv"))
    candidates.append("sector_map.csv")

    for path in candidates:
        if path and os.path.exists(path):
            try:
                df = pd.read_csv(path)
                if {"Ticker", "Sector"}.issubset(df.columns):
                    print(f"[attribution] Loaded sector map from: {path}")
                    return df[["Ticker", "Sector"]]
                else:
                    print(
                        f"[attribution] Sector map at {path} is missing "
                        "Ticker/Sector columns; skipping."
                    )
            except Exception as e:
                print(f"[attribution] Could not read sector map {path}: {e}")

    # No file found – try online build
    if not tickers:
        print("[attribution] No tickers provided to build sector map; skipping.")
        return None

    try:
        df = _auto_build_sector_map(tickers)
        # Cache for next run
        out_path = os.path.join(outdir, "sector_map.csv")
        try:
            df.to_csv(out_path, index=False)
            print(f"[attribution] Cached auto-built sector_map.csv to {out_path}")
        except Exception as e:
            print(f"[attribution] Could not write cached sector_map.csv: {e}")
        return df
    except Exception as e:
        print(f"[attribution] Failed to auto-build sector map: {e}")
        return None


# ------------------------------------------------------------
# Sector-level attribution helper
# ------------------------------------------------------------

def _run_sector_attribution(returns: pd.DataFrame,
                            wP_asset: pd.Series,
                            benchmark: str,
                            config: Dict,
                            outdir: str,
                            bench_total_return: float) -> None:
    """
    Optional sector-level Brinson–Fachler attribution.

    Uses:
      - sector_map.csv or config["sector_map_path"] if present,
        OR auto-builds sector map from yfinance.
      - Optional config keys:
            "sector_benchmarks": { "Tech": "XLK", ... }
            "benchmark_sector_weights": { "Tech": 0.28, ... }

    If optional keys are missing, it still computes sector attribution but
    allocation effects may be small/zero.
    """
    sector_map_df = _load_or_build_sector_map(config, outdir, list(wP_asset.index))
    if sector_map_df is None:
        return

    # Join weights with sectors
    weights_df = (
        pd.DataFrame({"wP": wP_asset})
        .join(sector_map_df.set_index("Ticker"), how="left")
    )

    missing_sector = weights_df["Sector"].isna()
    if missing_sector.any():
        missing_tickers = weights_df.index[missing_sector].tolist()
        print(
            "[attribution] Warning: some tickers have no sector mapping and "
            f"will be excluded from sector attribution: {missing_tickers}"
        )
        weights_df = weights_df[~missing_sector]

    if weights_df.empty:
        print(
            "[attribution] After filtering for sector mappings, "
            "no assets remain; skipping sector attribution."
        )
        return

    # Portfolio sector weights
    wP_sector = weights_df.groupby("Sector")["wP"].sum()
    if wP_sector.sum() == 0:
        print("[attribution] Sector weights sum to zero; skipping sector attribution.")
        return
    wP_sector = wP_sector / wP_sector.sum()

    sectors = wP_sector.index.tolist()

    # Sector-level portfolio cumulative returns
    rP_sector: Dict[str, float] = {}

    for sec in sectors:
        tickers_sec = weights_df.loc[weights_df["Sector"] == sec].index.tolist()
        tickers_sec = [t for t in tickers_sec if t in returns.columns]

        if not tickers_sec:
            print(
                f"[attribution] No return series found for sector {sec}; "
                "using 0 as sector return."
            )
            rP_sector[sec] = 0.0
            continue

        # Normalize weights within the sector
        w_in_sec = weights_df.loc[tickers_sec, "wP"]
        w_in_sec = w_in_sec / w_in_sec.sum()

        sub = returns[tickers_sec].copy().fillna(0.0)
        sec_monthly = (sub * w_in_sec.values).sum(axis=1)

        if sec_monthly.empty:
            rP_sector[sec] = 0.0
        else:
            rP_sector[sec] = float(np.prod(1 + sec_monthly) - 1)

    rP_sector = pd.Series(rP_sector)

    # Benchmark sector weights
    bmk_sector_weights_cfg = config.get("benchmark_sector_weights")
    if isinstance(bmk_sector_weights_cfg, dict) and bmk_sector_weights_cfg:
        wB_sector = pd.Series(bmk_sector_weights_cfg, dtype=float)
        wB_sector = wB_sector.reindex(sectors).fillna(0.0)
        if wB_sector.sum() > 0:
            wB_sector = wB_sector / wB_sector.sum()
        else:
            print(
                "[attribution] benchmark_sector_weights sums to zero; "
                "falling back to portfolio sector weights."
            )
            wB_sector = wP_sector.copy()
    else:
        wB_sector = wP_sector.copy()

    # Benchmark sector returns
    rB_sector: Dict[str, float] = {}
    sector_benchmarks_cfg = config.get("sector_benchmarks", {})

    for sec in sectors:
        proxy_ticker = None
        if isinstance(sector_benchmarks_cfg, dict):
            proxy_ticker = sector_benchmarks_cfg.get(sec)

        if proxy_ticker and proxy_ticker in returns.columns:
            s = returns[proxy_ticker].dropna()
            if not s.empty:
                rB_sector[sec] = float(np.prod(1 + s) - 1)
                continue

        # Fallback: use total benchmark return
        rB_sector[sec] = bench_total_return

    rB_sector = pd.Series(rB_sector)

    # Compute sector-level Brinson–Fachler
    try:
        df_sector = compute_brinson_fachler(
            wP=wP_sector, wB=wB_sector, rP_i=rP_sector, rB_i=rB_sector
        )
    except Exception as e:
        print(f"[attribution] Error computing sector attribution: {e}")
        return

    # Save sector CSV
    csv_path = os.path.join(outdir, "performance_attribution_sector.csv")
    try:
        df_sector.to_csv(csv_path, index_label="Sector")
        print(f"[attribution] Saved: {csv_path}")
    except Exception as e:
        print(f"[attribution] Could not write sector CSV: {e}")

    # Save sector plot
    png_path = os.path.join(outdir, "performance_attribution_sector.png")
    _plot_attribution(
        df_sector,
        png_path,
        title="Brinson–Fachler Performance Attribution (Sectors)",
    )


# ------------------------------------------------------------
# Main orchestrator
# ------------------------------------------------------------

def run_performance_attribution(outdir: str = "outputs",
                                config_path: str = "config.json") -> None:
    """
    Produces (if data available):
      - performance_attribution.csv              [asset-level]
      - performance_attribution.png
      - performance_attribution_sector.csv      [sector-level, optional]
      - performance_attribution_sector.png

    Returns nothing (safe failure allowed).
    """
    print("[attribution] Starting Performance Attribution...")

    # ---- LOAD CONFIG ----
    config = _load_config(config_path)
    if config is None:
        print("[attribution] No config; aborting attribution.")
        return

    benchmark = config.get("passive_benchmark") or config.get("benchmark")
    if benchmark is None:
        print("[attribution] Benchmark not specified; aborting.")
        return

    # ---- LOAD RETURNS ----
    ret_path = os.path.join(outdir, "monthly_returns.csv")
    returns = _safe_read_csv(ret_path)
    if returns is None:
        print("[attribution] No returns file.")
        return

    if "Date" in returns.columns:
        returns.set_index("Date", inplace=True)

    # ---- LOAD HOLDINGS ----
    holdings_path = os.path.join(outdir, "holdings_table.csv")
    holdings = _safe_read_csv(holdings_path)
    if holdings is None:
        print("[attribution] No holdings; aborting.")
        return

    if "Ticker" not in holdings.columns:
        print("[attribution] holdings_table.csv missing 'Ticker' column.")
        return

    # Prefer RealizedWeight -> TargetWeight -> Weight
    weight_col = None
    for candidate in ["RealizedWeight", "TargetWeight", "Weight"]:
        if candidate in holdings.columns:
            weight_col = candidate
            break

    if weight_col is None:
        print(
            "[attribution] holdings_table.csv has no usable weight column "
            "(expected one of RealizedWeight, TargetWeight, Weight)."
        )
        return

    print(f"[attribution] Using holdings weight column: {weight_col}")

    active_weights = holdings.set_index("Ticker")[weight_col].astype(float)

    if active_weights.sum() == 0:
        print("[attribution] Sum of active weights is zero; aborting.")
        return

    wP_asset = active_weights / active_weights.sum()

    # ---- Benchmark weights (equal-weight matching active universe) ----
    wB_asset = pd.Series(1.0 / len(wP_asset), index=wP_asset.index)

    # ---- Compute asset-level cumulative returns ----
    bucket_rP: Dict[str, float] = {}
    bucket_rB: Dict[str, float] = {}

    for t in wP_asset.index:
        if t not in returns.columns:
            print(f"[attribution] Warning: missing return series for {t}. Using 0.")
            bucket_rP[t] = 0.0
        else:
            r = returns[t].dropna()
            bucket_rP[t] = float(np.prod(1 + r) - 1) if not r.empty else 0.0

    if benchmark not in returns.columns:
        print(f"[attribution] Benchmark {benchmark} missing from returns.")
        return

    rB_series = returns[benchmark].dropna()
    bench_total_return = float(np.prod(1 + rB_series) - 1) if not rB_series.empty else 0.0

    for t in wP_asset.index:
        bucket_rB[t] = bench_total_return

    rP_asset = pd.Series(bucket_rP)
    rB_asset = pd.Series(bucket_rB)

    # ---- Asset-level Brinson–Fachler ----
    try:
        df_attr = compute_brinson_fachler(
            wP=wP_asset, wB=wB_asset, rP_i=rP_asset, rB_i=rB_asset
        )
    except Exception as e:
        print(f"[attribution] Error computing attribution: {e}")
        return

    # ---- Save CSV ----
    csv_path = os.path.join(outdir, "performance_attribution.csv")
    try:
        df_attr.to_csv(csv_path, index_label="Bucket")
        print(f"[attribution] Saved: {csv_path}")
    except Exception as e:
        print(f"[attribution] Could not write CSV: {e}")

    # ---- Save Plot ----
    png_path = os.path.join(outdir, "performance_attribution.png")
    _plot_attribution(df_attr, png_path)

    # ---- Sector-level attribution (optional) ----
    try:
        _run_sector_attribution(
            returns=returns,
            wP_asset=wP_asset,
            benchmark=benchmark,
            config=config,
            outdir=outdir,
            bench_total_return=bench_total_return,
        )
    except Exception as e:
        print(f"[attribution] Warning: sector-level attribution failed: {e}")

    print("[attribution] Done.")
