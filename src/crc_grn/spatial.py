from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import kneighbors_graph, NearestNeighbors


def build_knn(coords: pd.DataFrame, n_neighbors: int = 6):
    arr = coords[['x', 'y']].astype(float).values
    n = arr.shape[0]
    if n == 0:
        return csr_matrix((0, 0), dtype=float)
    if n == 1:
        return csr_matrix((1, 1), dtype=float)
    n_neighbors = max(1, min(int(n_neighbors), n - 1))
    return kneighbors_graph(arr, n_neighbors=n_neighbors, mode='connectivity', include_self=False)


def build_weighted_knn(coords: pd.DataFrame, n_neighbors: int = 6, sigma: float | None = None):
    """Distance-weighted KNN graph for spatial exposure calculations.

    We keep ``build_knn`` as the backward-compatible binary graph and add this weighted
    graph for cell-cell/spatial exposure models. Edge weights are Gaussian in distance.
    """
    arr = coords[['x', 'y']].astype(float).values
    n = arr.shape[0]
    if n == 0:
        return csr_matrix((0, 0), dtype=float)
    if n == 1:
        return csr_matrix((1, 1), dtype=float)
    k = max(1, min(int(n_neighbors), n - 1))
    nn = NearestNeighbors(n_neighbors=k + 1)
    nn.fit(arr)
    dist, ind = nn.kneighbors(arr)
    # Remove self neighbor in column 0.
    dist = dist[:, 1:]
    ind = ind[:, 1:]
    if sigma is None:
        positive = dist[dist > 0]
        sigma = float(np.nanmedian(positive)) if positive.size else 1.0
    sigma = max(float(sigma), 1e-6)
    weights = np.exp(-((dist.astype(float) ** 2) / (2.0 * sigma ** 2)))
    rows = np.repeat(np.arange(n), k)
    cols = ind.reshape(-1)
    vals = weights.reshape(-1)
    return csr_matrix((vals, (rows, cols)), shape=(n, n), dtype=float)


def spatial_lag(expr: pd.DataFrame, knn) -> pd.DataFrame:
    if expr.shape[0] == 0:
        return expr.copy()
    if getattr(knn, 'shape', None) == (expr.shape[0], expr.shape[0]) and knn.nnz == 0:
        return pd.DataFrame(np.zeros_like(expr.values, dtype=float), index=expr.index, columns=expr.columns)
    arr = knn.dot(expr.values)
    denom = np.asarray(knn.sum(axis=1)).reshape(-1, 1)
    denom[denom == 0] = 1.0
    lag = arr / denom
    return pd.DataFrame(lag, index=expr.index, columns=expr.columns)


def multi_scale_spatial_lag(expr: pd.DataFrame, coords: pd.DataFrame, ks: tuple[int, ...] = (6, 12, 24), weighted: bool = True) -> dict[int, pd.DataFrame]:
    """Return spatial lag matrices across neighborhood scales."""
    out: dict[int, pd.DataFrame] = {}
    for k in ks:
        if len(coords) <= 1:
            graph = csr_matrix((len(coords), len(coords)), dtype=float)
        elif weighted:
            graph = build_weighted_knn(coords, n_neighbors=min(int(k), len(coords) - 1))
        else:
            graph = build_knn(coords, n_neighbors=min(int(k), len(coords) - 1))
        out[int(k)] = spatial_lag(expr, graph)
    return out
