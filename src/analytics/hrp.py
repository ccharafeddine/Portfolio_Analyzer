"""
Hierarchical Risk Parity (HRP) portfolio construction.

Implements the Lopez de Prado (2016) algorithm:
1. Compute correlation-distance matrix
2. Hierarchical clustering via single-linkage
3. Quasi-diagonalize the covariance matrix
4. Recursive bisection to assign weights
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform


def _correlation_distance(returns: pd.DataFrame) -> np.ndarray:
    """Convert correlation matrix to a distance matrix."""
    corr = returns.corr()
    dist = ((1 - corr) / 2.0) ** 0.5
    return dist


def _quasi_diagonalize(link: np.ndarray) -> list[int]:
    """Reorder assets to quasi-diagonalize the covariance matrix."""
    n = int(link[-1, 3])
    sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
    num_items = link[-1, 3]

    while sort_ix.max() >= n:
        sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
        df0 = sort_ix[sort_ix >= n]
        i = df0.index
        j = df0.values.astype(int) - n
        sort_ix[i] = link[j, 0]
        df1 = pd.Series(link[j, 1], index=i + 1)
        sort_ix = pd.concat([sort_ix, df1])
        sort_ix = sort_ix.sort_index()
        sort_ix.index = range(sort_ix.shape[0])

    return sort_ix.astype(int).tolist()


def _cluster_var(cov: np.ndarray, cluster_items: list[int]) -> float:
    """Compute inverse-variance portfolio variance for a cluster."""
    cov_slice = cov[np.ix_(cluster_items, cluster_items)]
    ivp = 1.0 / np.diag(cov_slice)
    ivp /= ivp.sum()
    return float(ivp @ cov_slice @ ivp)


def _recursive_bisection(
    cov: np.ndarray,
    sorted_ix: list[int],
) -> np.ndarray:
    """Assign weights via recursive bisection."""
    n = cov.shape[0]
    w = np.ones(n)
    cluster_items = [sorted_ix]

    while len(cluster_items) > 0:
        # Split each cluster in half
        new_clusters = []
        for cluster in cluster_items:
            if len(cluster) <= 1:
                continue
            half = len(cluster) // 2
            left = cluster[:half]
            right = cluster[half:]

            var_left = _cluster_var(cov, left)
            var_right = _cluster_var(cov, right)

            alpha = 1.0 - var_left / (var_left + var_right)

            for i in left:
                w[i] *= alpha
            for i in right:
                w[i] *= (1.0 - alpha)

            if len(left) > 1:
                new_clusters.append(left)
            if len(right) > 1:
                new_clusters.append(right)

        cluster_items = new_clusters

    return w


def hrp_linkage_matrix(returns: pd.DataFrame) -> np.ndarray:
    """
    Compute the hierarchical clustering linkage matrix.

    Parameters
    ----------
    returns : asset returns DataFrame

    Returns
    -------
    linkage matrix (n-1, 4) array
    """
    dist = _correlation_distance(returns)
    dist_arr = dist.values.copy()
    np.fill_diagonal(dist_arr, 0)
    # Ensure perfect symmetry for squareform
    dist_arr = (dist_arr + dist_arr.T) / 2.0
    condensed = squareform(dist_arr)
    return linkage(condensed, method="single")


def hrp_weights(returns: pd.DataFrame) -> pd.Series:
    """
    Compute HRP portfolio weights.

    Parameters
    ----------
    returns : asset returns DataFrame (columns = assets)

    Returns
    -------
    pd.Series of weights summing to 1.0
    """
    if returns.shape[1] < 2:
        return pd.Series(1.0, index=returns.columns)

    cov = returns.cov().values
    link = hrp_linkage_matrix(returns)
    sorted_ix = _quasi_diagonalize(link)
    w = _recursive_bisection(cov, sorted_ix)

    # Normalize to sum to 1
    w = w / w.sum()

    return pd.Series(w, index=returns.columns, name="HRP_weight")
