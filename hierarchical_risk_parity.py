import os
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform


def _correlation_distance(corr: pd.DataFrame) -> np.ndarray:
    """
    López de Prado-style correlation distance matrix.

    d_ij = sqrt(0.5 * (1 - corr_ij))
    """
    c = corr.values.astype(float)
    # Ensure the matrix is in [-1, 1]
    c = np.clip(c, -1.0, 1.0)
    dist = np.sqrt(0.5 * (1.0 - c))
    return dist


def _quasi_diagonal_order(linkage_matrix: np.ndarray, labels: list[str]) -> list[str]:
    """
    Obtain a quasi-diagonal leaf ordering from the hierarchical tree.

    Uses scipy's dendrogram (no plot) to get an ordered list of leaf indices,
    then maps them back to the original labels.
    """
    dendro = hierarchy.dendrogram(linkage_matrix, no_plot=True)
    leaf_order = dendro["leaves"]
    return [labels[i] for i in leaf_order]


def _get_cluster_var(cov: pd.DataFrame, tickers: list[str]) -> float:
    """
    Compute cluster variance using inverse-variance portfolio within the cluster.
    """
    sub_cov = cov.loc[tickers, tickers].values.astype(float)
    # Guard against numerical issues
    diag = np.diag(sub_cov)
    # If any variances are non-positive, fall back to equal weights
    if np.any(diag <= 0):
        w = np.repeat(1.0 / len(tickers), len(tickers))
    else:
        iv = 1.0 / diag
        w = iv / iv.sum()
    cluster_var = float(w.T @ sub_cov @ w)
    return cluster_var


def _recursive_bisection(cov: pd.DataFrame, ordered_tickers: list[str]) -> dict[str, float]:
    """
    Recursively allocate risk across the ordered tickers.

    At each split, capital is allocated to left/right sub-clusters in inverse
    proportion to their cluster variances.
    """
    if len(ordered_tickers) == 1:
        return {ordered_tickers[0]: 1.0}

    # Split the list into two halves
    mid = len(ordered_tickers) // 2
    left = ordered_tickers[:mid]
    right = ordered_tickers[mid:]

    # Recurse on sub-clusters
    w_left = _recursive_bisection(cov, left)
    w_right = _recursive_bisection(cov, right)

    # Compute cluster variances
    var_left = _get_cluster_var(cov, list(w_left.keys()))
    var_right = _get_cluster_var(cov, list(w_right.keys()))
    total = var_left + var_right
    if total <= 0:
        alpha_left = alpha_right = 0.5
    else:
        # lower variance cluster gets higher allocation
        alpha_left = 1.0 - var_left / total
        alpha_right = 1.0 - alpha_left

    # Scale sub-weights by cluster allocations
    weights: dict[str, float] = {}
    for k, v in w_left.items():
        weights[k] = float(v * alpha_left)
    for k, v in w_right.items():
        weights[k] = float(v * alpha_right)

    return weights


def compute_hrp_weights(
    asset_returns: pd.DataFrame,
    outdir: Optional[str] = None,
    linkage_method: str = "single",
) -> pd.Series:
    """
    Compute Hierarchical Risk Parity (HRP) weights from asset return series.

    Parameters
    ----------
    asset_returns : DataFrame
        Monthly (or other frequency) returns of the assets in the active universe.
        Columns are tickers, index is DatetimeIndex.
    outdir : str, optional
        If provided, hrp_weights.csv will be written into this folder.
    linkage_method : str
        Linkage method for scipy.cluster.hierarchy.linkage (default 'single').

    Returns
    -------
    Series
        HRP weights indexed by ticker, summing to 1.0.
    """
    # Basic cleaning
    rets = asset_returns.dropna(how="all").copy()
    rets = rets.dropna(axis=1, how="all")

    if rets.shape[1] < 2:
        # Not enough assets for clustering; fallback to 100% in the single asset
        if rets.shape[1] == 1:
            w = pd.Series([1.0], index=rets.columns, name="weight")
        else:
            w = pd.Series(dtype="float64", name="weight")
        if outdir is not None:
            os.makedirs(outdir, exist_ok=True)
            w.to_frame(name="weight").to_csv(os.path.join(outdir, "hrp_weights.csv"))
        return w

    cov = rets.cov()
    corr = rets.corr()
    labels = list(rets.columns)

    # Build distance matrix and hierarchical tree
    dist = _correlation_distance(corr)
    condensed = squareform(dist, checks=False)
    link = hierarchy.linkage(condensed, method=linkage_method)

    ordered = _quasi_diagonal_order(link, labels)

    raw_weights = _recursive_bisection(cov, ordered)
    w = pd.Series(raw_weights, name="weight")

    # Normalize to sum exactly to 1.0
    total = float(w.sum())
    if total != 0.0:
        w = w / total

    # Optional: save to CSV
    if outdir is not None:
        os.makedirs(outdir, exist_ok=True)
        df_out = w.to_frame(name="weight")
        df_out["weight_pct"] = df_out["weight"] * 100.0
        df_out = df_out.sort_values("weight", ascending=False)
        df_out.to_csv(os.path.join(outdir, "hrp_weights.csv"))

    return w


def plot_hrp_dendrogram(
    asset_returns: pd.DataFrame,
    outdir: str,
    linkage_method: str = "single",
    fname: str = "hrp_cluster_tree.png",
) -> None:
    """
    Generate and save a dendrogram (cluster tree) for the asset correlation structure.
    """
    rets = asset_returns.dropna(how="all").copy()
    rets = rets.dropna(axis=1, how="all")

    if rets.shape[1] < 2:
        # Nothing to cluster
        return

    corr = rets.corr()
    labels = list(rets.columns)

    dist = _correlation_distance(corr)
    condensed = squareform(dist, checks=False)
    link = hierarchy.linkage(condensed, method=linkage_method)

    os.makedirs(outdir, exist_ok=True)
    plt.figure(figsize=(10, 6))
    hierarchy.dendrogram(link, labels=labels, leaf_rotation=90)
    plt.title("Hierarchical Clustering Tree (Correlation-Based)")
    plt.tight_layout()
    out_path = os.path.join(outdir, fname)
    plt.savefig(out_path, dpi=300)
    plt.close()
