from __future__ import annotations

import numpy as np
import pandas as pd


def _is_count_like(values: np.ndarray) -> bool:
    vals = values[np.isfinite(values)]
    if vals.size == 0:
        return False
    if float(np.nanmax(vals)) <= 30 and np.mean(np.abs(vals - np.round(vals)) < 1e-8) < 0.80:
        return False
    return float(np.nanmax(vals)) > 30 or np.mean(np.abs(vals - np.round(vals)) < 1e-8) >= 0.80


def normalize_expression_matrix(
    expr: pd.DataFrame,
    target_sum: float = 1e4,
    assume_logged: bool | None = None,
) -> pd.DataFrame:
    """Return a non-negative log1p(CPM) expression matrix.

    The package previously mixed raw counts, already-normalized matrices and sample-wise
    matrices. That makes spatial exposure and GRN scores depend on library size. This
    helper is intentionally conservative: if a matrix looks already log-normalized it is
    preserved; count-like matrices are library-size normalized and log1p transformed.
    """
    if expr is None or expr.empty:
        return pd.DataFrame(index=getattr(expr, 'index', None), columns=getattr(expr, 'columns', None))
    x = expr.copy()
    x.columns = x.columns.astype(str)
    x = x.apply(pd.to_numeric, errors='coerce').fillna(0.0)
    x[x < 0] = 0.0
    arr = x.values.astype(float, copy=False)
    if assume_logged is None:
        assume_logged = not _is_count_like(arr)
    if assume_logged:
        return x.astype(float)
    lib = arr.sum(axis=1)
    lib[~np.isfinite(lib) | (lib <= 0)] = 1.0
    norm = np.log1p((arr / lib[:, None]) * float(target_sum))
    norm[~np.isfinite(norm)] = 0.0
    return pd.DataFrame(norm, index=x.index, columns=x.columns)


def robust_gene_filter(
    expr: pd.DataFrame,
    min_nonzero_fraction: float = 0.005,
    min_variance: float = 1e-8,
) -> pd.Series:
    """Boolean gene mask using detection rate and variance."""
    if expr.empty:
        return pd.Series(dtype=bool)
    var = expr.var(axis=0)
    nz = (expr > 0).mean(axis=0)
    keep = (var > float(min_variance)) & (nz >= float(min_nonzero_fraction))
    return keep.fillna(False).astype(bool)


def samplewise_rank_transform(expr: pd.DataFrame) -> pd.DataFrame:
    """Rank-normalize rows for optional robust downstream visual diagnostics."""
    if expr.empty:
        return expr.copy()
    return expr.rank(axis=1, pct=True).fillna(0.0).astype(float)

# MODIFIED: Add sample-aware batch residualization and robust HVG helpers to prevent pooled CRC/NAT or multi-sample artifacts from dominating GRN scores.
def residualize_batch_effect(expr: pd.DataFrame, batch: pd.Series | list | np.ndarray | None, min_cells_per_batch: int = 3) -> pd.DataFrame:
    """Return expression after subtracting gene-wise batch offsets.

    This is a deliberately light-weight ComBat-like centering step for the pooled
    scoring layer. It preserves the global gene mean but removes sample-specific
    shifts, so downstream pooled correlations are less driven by sequencing depth,
    tissue handling, or sample identity. Within-sample scoring remains untouched.
    """
    if expr is None or expr.empty or batch is None:
        return expr.copy() if isinstance(expr, pd.DataFrame) else expr
    out = expr.astype(float).copy()
    b = pd.Series(batch, index=out.index).astype(str)
    global_mean = out.mean(axis=0)
    corrected = out.copy()
    for label, idx in b.groupby(b).groups.items():
        idx = list(idx)
        if len(idx) < int(min_cells_per_batch):
            continue
        local_mean = out.loc[idx].mean(axis=0)
        corrected.loc[idx] = out.loc[idx] - local_mean + global_mean
    corrected[corrected < 0] = 0.0
    corrected[~np.isfinite(corrected)] = 0.0
    return corrected


def select_highly_variable_genes(expr: pd.DataFrame, top_n: int = 1200, min_detection: float = 0.01) -> list[str]:
    """Select robust HVGs by dispersion after filtering very sparse genes."""
    if expr is None or expr.empty:
        return []
    x = expr.astype(float)
    det = (x > 0).mean(axis=0)
    mean = x.mean(axis=0)
    var = x.var(axis=0)
    dispersion = var / (mean + 1e-8)
    rank_df = pd.DataFrame({'gene': x.columns.astype(str), 'detection': det.values, 'dispersion': dispersion.values, 'mean': mean.values})
    rank_df = rank_df[(rank_df['detection'] >= float(min_detection)) & np.isfinite(rank_df['dispersion'])].copy()
    if rank_df.empty:
        return []
    return rank_df.sort_values(['dispersion', 'mean'], ascending=[False, False])['gene'].head(int(top_n)).astype(str).tolist()
# END MODIFIED

