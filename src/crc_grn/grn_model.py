from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_corr(a, b) -> float:
    x = pd.Series(a, copy=False).astype(float)
    y = pd.Series(b, copy=False).astype(float)
    mask = x.notna() & y.notna()
    if mask.sum() < 3:
        return 0.0
    xv = x[mask].values
    yv = y[mask].values
    if np.nanstd(xv) <= 1e-12 or np.nanstd(yv) <= 1e-12:
        return 0.0
    c = np.corrcoef(xv, yv)[0, 1]
    return 0.0 if not np.isfinite(c) else float(c)


def _safe_zscore(v: np.ndarray) -> np.ndarray:
    arr = np.asarray(v, dtype=float)
    if arr.size == 0:
        return arr.astype(float)
    mu = np.nanmean(arr)
    sd = np.nanstd(arr)
    if not np.isfinite(sd) or sd <= 1e-12:
        return np.zeros_like(arr, dtype=float)
    out = (arr - mu) / sd
    out[~np.isfinite(out)] = 0.0
    return out.astype(float)


def _clip01(v: float, upper: float = 1.0) -> float:
    if not np.isfinite(v):
        return 0.0
    return float(max(0.0, min(float(v), upper)))


def _as_2d_cov(cov: np.ndarray | None, n: int) -> np.ndarray:
    if cov is None:
        return np.zeros((n, 0), dtype=float)
    arr = np.asarray(cov, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.shape[0] != n:
        return np.zeros((n, 0), dtype=float)
    if arr.shape[1] == 0:
        return np.zeros((n, 0), dtype=float)
    arr = np.apply_along_axis(_safe_zscore, 0, arr)
    arr[~np.isfinite(arr)] = 0.0
    # Remove zero-variance covariates to keep least-squares stable.
    keep = np.nanstd(arr, axis=0) > 1e-12
    return arr[:, keep] if np.any(keep) else np.zeros((n, 0), dtype=float)


def _fit_r2(design: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, float]:
    y = _safe_zscore(target)
    n = len(y)
    if n < 3:
        return np.zeros(1, dtype=float), 0.0
    X = np.asarray(design, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    if X.shape[0] != n:
        return np.zeros(1, dtype=float), 0.0
    if X.shape[1] > 0:
        X = np.apply_along_axis(_safe_zscore, 0, X)
    X = np.c_[np.ones(n), X]
    try:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        pred = X @ beta
        resid = y - pred
        sst = float(np.sum((y - y.mean()) ** 2))
        sse = float(np.sum(resid ** 2))
        r2 = 0.0 if sst <= 1e-12 else max(0.0, min(1.0, 1.0 - sse / sst))
        return beta, float(r2)
    except Exception:
        return np.zeros(X.shape[1], dtype=float), 0.0


def _residualize_multi(a: np.ndarray, cov: np.ndarray | None) -> np.ndarray:
    a = _safe_zscore(a)
    n = len(a)
    if n < 3:
        return np.zeros_like(a, dtype=float)
    cov2 = _as_2d_cov(cov, n)
    if cov2.shape[1] == 0:
        return a
    design = np.c_[np.ones(n), cov2]
    try:
        beta, *_ = np.linalg.lstsq(design, a, rcond=None)
        resid = a - design @ beta
    except Exception:
        resid = np.zeros_like(a, dtype=float)
    resid[~np.isfinite(resid)] = 0.0
    return resid.astype(float)


def _partial_corr_multi(a: np.ndarray, b: np.ndarray, controls: np.ndarray | None) -> float:
    ra = _residualize_multi(a, controls)
    rb = _residualize_multi(b, controls)
    return _safe_corr(ra, rb)


def _ols_delta_stats(x: np.ndarray, z: np.ndarray, y: np.ndarray, cov: np.ndarray | None = None) -> dict:
    """Estimate internal/external contributions conditional on local covariates.

    The baseline model contains only local/background covariates. The internal model adds
    same-spot TF expression, the external model adds the relative neighborhood exposure,
    and the full model includes both. Delta-R2 terms are computed against the opposite
    branch so the external effect is the additional variance explained beyond TF/covariates.
    """
    x = _safe_zscore(x)
    z = _safe_zscore(z)
    y = _safe_zscore(y)
    n = len(y)
    cov2 = _as_2d_cov(cov, n)
    if n < 6:
        return {
            'internal_beta': 0.0,
            'external_beta': 0.0,
            'full_r2': 0.0,
            'r2_covariates_only': 0.0,
            'r2_internal_only': 0.0,
            'r2_external_only': 0.0,
            'internal_delta_r2': 0.0,
            'external_delta_r2': 0.0,
        }

    _, r2_cov = _fit_r2(cov2, y)
    beta_x, r2_x = _fit_r2(np.c_[cov2, x], y)
    beta_z, r2_z = _fit_r2(np.c_[cov2, z], y)
    beta_full, r2_full = _fit_r2(np.c_[cov2, x, z], y)
    # beta order: intercept, covariates..., x, z
    x_beta_pos = 1 + cov2.shape[1]
    z_beta_pos = x_beta_pos + 1
    return {
        'internal_beta': float(beta_full[x_beta_pos]) if len(beta_full) > x_beta_pos else 0.0,
        'external_beta': float(beta_full[z_beta_pos]) if len(beta_full) > z_beta_pos else 0.0,
        'full_r2': float(r2_full),
        'r2_covariates_only': float(r2_cov),
        'r2_internal_only': float(r2_x),
        'r2_external_only': float(r2_z),
        'internal_delta_r2': float(max(r2_full - r2_z, 0.0)),
        'external_delta_r2': float(max(r2_full - r2_x, 0.0)),
    }


def _directional_value(value: float, regulatory_sign: int) -> float:
    """Return evidence in the biologically expected TF-target direction.

    +1 expects positive association, -1 expects inverse association and 0 treats both
    activation and repression as plausible. This avoids systematically discarding
    repressive TF-target edges, a common cause of flat GRN AUC/F1 curves.
    """
    v = 0.0 if not np.isfinite(value) else float(value)
    sign = int(regulatory_sign) if np.isfinite(regulatory_sign) else 0
    if sign > 0:
        return max(v, 0.0)
    if sign < 0:
        return max(-v, 0.0)
    return abs(v)


def _blend_external_signal(
    tf: str,
    tg: str,
    lag_expr: pd.DataFrame,
    multi_lag_expr: dict[int, pd.DataFrame] | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """Build a delayed/cascaded neighborhood exposure feature.

    The original code used one spatial lag only. In spatial transcriptomics that makes
    the model overly sensitive to a single KNN radius and misses delayed diffusion-like
    signaling. We use the supplied multiscale lags as near/mid/far pseudo-time steps and
    weight them as a decaying cascade. If multiscale lags are absent, behavior falls back
    to the original single-lag design.
    """
    def _gene_vec(df: pd.DataFrame, gene: str, fallback_len: int) -> np.ndarray:
        if gene in df.columns:
            return df[gene].astype(float).values
        return np.zeros(fallback_len, dtype=float)

    n = lag_expr.shape[0]
    if not multi_lag_expr:
        lag_tg = _gene_vec(lag_expr, tg, n)
        lag_tf = _gene_vec(lag_expr, tf, n)
        z = 0.70 * _safe_zscore(lag_tg) + 0.30 * _safe_zscore(lag_tf)
        return z, {'cascade_gain': 1.0, 'n_scales': 1.0}

    scales = sorted(int(k) for k in multi_lag_expr if isinstance(multi_lag_expr[k], pd.DataFrame) and multi_lag_expr[k].shape[0] == n)
    if not scales:
        lag_tg = _gene_vec(lag_expr, tg, n)
        lag_tf = _gene_vec(lag_expr, tf, n)
        z = 0.70 * _safe_zscore(lag_tg) + 0.30 * _safe_zscore(lag_tf)
        return z, {'cascade_gain': 1.0, 'n_scales': 1.0}

    # Decay over spatial radius: near neighbors dominate, farther neighbors model
    # delayed/cascade exposure. Normalize to keep the scale stable across datasets.
    raw_w = np.exp(-0.55 * np.arange(len(scales), dtype=float))
    raw_w = raw_w / raw_w.sum()
    parts = []
    for w, k in zip(raw_w, scales):
        df = multi_lag_expr[k]
        lag_tg = _gene_vec(df, tg, n)
        lag_tf = _gene_vec(df, tf, n)
        parts.append(float(w) * (0.70 * _safe_zscore(lag_tg) + 0.30 * _safe_zscore(lag_tf)))
    z = np.sum(parts, axis=0)
    # Amplification is high when the multiscale signals are coherent rather than noisy.
    corr_vals = []
    for i in range(len(parts)):
        for j in range(i + 1, len(parts)):
            corr_vals.append(abs(_safe_corr(parts[i], parts[j])))
    cascade_gain = 1.0 + 0.25 * (float(np.nanmean(corr_vals)) if corr_vals else 0.0)
    return z * cascade_gain, {'cascade_gain': cascade_gain, 'n_scales': float(len(scales))}


def edge_scores(
    expr: pd.DataFrame,
    lag_expr: pd.DataFrame,
    candidate_edges: pd.DataFrame,
    multi_lag_expr: dict[int, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    rows = []
    has_prior_conf = 'prior_conf' in candidate_edges.columns
    empty_cols = [
        'source', 'target', 'prior', 'regulatory_sign', 'internal', 'external', 'external_abs', 'external_relative',
        'internal_support', 'external_support', 'directional_internal_support', 'directional_corr_support',
        'internal_delta_r2', 'external_delta_r2', 'external_specificity', 'external_direction',
        'corr_xy', 'corr_yz', 'corr_xz', 'full_r2', 'r2_covariates_only', 'r2_internal_only',
        'r2_external_only', 'partial_xy_z', 'partial_yz_x', 'neighborhood_exposure_mean_abs',
        'neighborhood_exposure_abs_mean_abs', 'cascade_gain', 'n_external_scales', 'attention_weight', 'internal_purity', 'external_confounding_penalty', 'intrinsic_score', 'external_branch_score', 'score'
    ]
    if expr.empty or lag_expr.empty or candidate_edges.empty:
        return pd.DataFrame(columns=empty_cols)

    expr = expr.copy()
    lag_expr = lag_expr.copy()
    expr.columns = expr.columns.astype(str)
    lag_expr.columns = lag_expr.columns.astype(str)

    # Local/background covariates control for library burden and broad niche-level expression.
    expr_burden = _safe_zscore(expr.astype(float).sum(axis=1).values)
    lag_burden = _safe_zscore(lag_expr.astype(float).sum(axis=1).values)

    for _, r in candidate_edges.iterrows():
        tf = str(r['source'])
        tg = str(r['target'])
        if tf == tg:
            continue
        if tf not in expr.columns or tg not in expr.columns or tg not in lag_expr.columns:
            continue
        x = expr[tf].astype(float).values
        y = expr[tg].astype(float).values
        if len(x) < 6 or np.nanstd(x) <= 1e-12 or np.nanstd(y) <= 1e-12:
            continue

        z_abs, cascade_meta = _blend_external_signal(tf, tg, lag_expr, multi_lag_expr)
        local_cov = np.c_[expr_burden, lag_burden]
        # Relative external effect: neighborhood exposure left after removing same-spot TF
        # and broad local expression/background covariates.
        z_rel = _residualize_multi(z_abs, np.c_[x, local_cov])

        stats = _ols_delta_stats(x, z_rel, y, cov=local_cov)
        corr_xy = _safe_corr(x, y)
        corr_xz = _safe_corr(x, z_rel)
        corr_yz = _safe_corr(y, z_rel)
        partial_xy_z = _partial_corr_multi(x, y, np.c_[z_rel, local_cov])
        partial_yz_x = _partial_corr_multi(z_rel, y, np.c_[x, local_cov])

        internal = float(partial_xy_z)
        external = float(partial_yz_x)
        sign = int(r.get('regulatory_sign', 0)) if pd.notna(r.get('regulatory_sign', 0)) else 0
        # MODIFIED: Treat spatial external signal as a confounder for GRN edge ranking,
        # not as automatic positive evidence. The previous formula rewarded target spatial
        # autocorrelation, causing many edges to be ranked by niche smoothness rather than
        # TF-target regulation. Internal GRN ranking now emphasizes direction-aware
        # same-spot support and explicitly penalizes external-only explanations.
        external_specificity = max(0.0, 1.0 - abs(corr_xz))
        internal_delta = float(stats['internal_delta_r2'])
        external_delta = float(stats['external_delta_r2'])
        directional_internal = _directional_value(internal, sign)
        directional_corr = _directional_value(corr_xy, sign)
        internal_purity = float(internal_delta / (internal_delta + external_delta + 1e-9))
        raw_internal_support = directional_internal * internal_delta
        external_branch_score = max(float(stats['external_beta']), 0.0) * external_delta * external_specificity
        external_confounding_penalty = external_branch_score * (1.0 - min(1.0, 2.0 * directional_internal))
        internal_support = raw_internal_support * (0.50 + 0.50 * internal_purity)
        external_support = external_branch_score
        prior = float(r['prior_conf']) if has_prior_conf and pd.notna(r['prior_conf']) else 1.0
        prior_term = min(max(prior, 0.0), 3.0) / 3.0
        att_raw = 0.90 * prior_term + 0.85 * directional_internal + 0.45 * internal_purity - 0.35 * external_confounding_penalty
        attention_weight = float(1.0 / (1.0 + np.exp(-att_raw)))
        intrinsic_score = attention_weight * (
            0.26 * prior_term
            + 0.38 * internal_support
            + 0.14 * directional_internal
            + 0.10 * directional_corr
            + 0.08 * _clip01(stats['full_r2']) * internal_purity
            + 0.04 * external_specificity * internal_purity
        )
        score = float(max(0.0, intrinsic_score - 0.18 * external_confounding_penalty))
        # END MODIFIED
        rows.append({
            'source': tf,
            'target': tg,
            'prior': prior,
            'regulatory_sign': sign,
            'internal': internal,
            'external': external,
            'external_abs': float(_safe_corr(z_abs, y)),
            'external_relative': external,
            'internal_beta': float(stats['internal_beta']),
            'external_beta': float(stats['external_beta']),
            'internal_support': float(internal_support),
            'external_support': float(external_support),
            'directional_internal_support': float(directional_internal),
            'directional_corr_support': float(directional_corr),
            'internal_delta_r2': internal_delta,
            'external_delta_r2': external_delta,
            'external_specificity': float(external_specificity),
            'external_direction': 'positive' if float(stats['external_beta']) > 0 else ('negative' if float(stats['external_beta']) < 0 else 'flat'),
            'corr_xy': float(corr_xy),
            'corr_yz': float(corr_yz),
            'corr_xz': float(corr_xz),
            'full_r2': float(stats['full_r2']),
            'r2_covariates_only': float(stats['r2_covariates_only']),
            'r2_internal_only': float(stats['r2_internal_only']),
            'r2_external_only': float(stats['r2_external_only']),
            'partial_xy_z': float(partial_xy_z),
            'partial_yz_x': float(partial_yz_x),
            'neighborhood_exposure_mean_abs': float(np.mean(np.abs(z_rel))),
            'neighborhood_exposure_abs_mean_abs': float(np.mean(np.abs(z_abs))),
            'cascade_gain': float(cascade_meta.get('cascade_gain', 1.0)),
            'n_external_scales': float(cascade_meta.get('n_scales', 1.0)),
            'attention_weight': attention_weight,
            # MODIFIED: expose internal purity and external-only penalty for diagnostics and downstream aggregation.
            'internal_purity': float(internal_purity),
            'external_confounding_penalty': float(external_confounding_penalty),
            'intrinsic_score': float(intrinsic_score),
            'external_branch_score': float(external_branch_score),
            # END MODIFIED
            'score': float(score),
        })
    cols = empty_cols[:-1]
    # Add columns generated above that are not in the empty-list display order.
    cols = [
        'source', 'target', 'prior', 'regulatory_sign', 'internal', 'external', 'external_abs', 'external_relative',
        'internal_beta', 'external_beta', 'internal_support', 'external_support',
        'directional_internal_support', 'directional_corr_support',
        'internal_delta_r2', 'external_delta_r2', 'external_specificity', 'external_direction',
        'corr_xy', 'corr_yz', 'corr_xz', 'full_r2', 'r2_covariates_only', 'r2_internal_only',
        'r2_external_only', 'partial_xy_z', 'partial_yz_x', 'neighborhood_exposure_mean_abs',
        'neighborhood_exposure_abs_mean_abs', 'cascade_gain', 'n_external_scales', 'attention_weight', 'internal_purity', 'external_confounding_penalty', 'intrinsic_score', 'external_branch_score', 'score'
    ]
    if not rows:
        return pd.DataFrame(columns=cols)
    out = pd.DataFrame(rows)
    return out.sort_values(
        ['score', 'internal_support', 'external_support', 'full_r2', 'prior'],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)
