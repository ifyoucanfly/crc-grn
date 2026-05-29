from __future__ import annotations

import hashlib
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, fisher_exact, combine_pvalues

from .spatial import build_weighted_knn, spatial_lag


def _safe_zscore(v: np.ndarray | pd.Series) -> np.ndarray:
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


def _safe_spearman(a, b) -> tuple[float, float]:
    x = pd.Series(a, copy=False).astype(float)
    y = pd.Series(b, copy=False).astype(float)
    mask = x.notna() & y.notna()
    if int(mask.sum()) < 4:
        return 0.0, 1.0
    xv = x[mask].values
    yv = y[mask].values
    if np.nanstd(xv) <= 1e-12 or np.nanstd(yv) <= 1e-12:
        return 0.0, 1.0
    try:
        rho, p = spearmanr(xv, yv)
        rho = 0.0 if not np.isfinite(rho) else float(rho)
        p = 1.0 if not np.isfinite(p) else float(p)
        return rho, p
    except Exception:
        return 0.0, 1.0


def _bh_fdr(pvalues: Iterable[float]) -> np.ndarray:
    p = np.asarray(list(pvalues), dtype=float)
    if p.size == 0:
        return p
    p[~np.isfinite(p)] = 1.0
    p = np.clip(p, 0.0, 1.0)
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * len(p) / (np.arange(len(p)) + 1.0)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)
    out = np.empty_like(q)
    out[order] = q
    return out


def _signature_id(ligand: str, receptor: str, sender_niche: str = 'ANY', receiver_niche: str = 'ANY') -> str:
    core = f'{ligand}|{receptor}|{sender_niche}|{receiver_niche}'
    h = hashlib.sha1(core.encode('utf-8')).hexdigest()[:10]
    return f'CCC_{h}'


def _lr_pair_id(ligand: str, receptor: str) -> str:
    return _signature_id(ligand, receptor, 'ANY', 'ANY')


def infer_spatial_lr_edges(
    expr: pd.DataFrame,
    coords: pd.DataFrame,
    lr_prior: pd.DataFrame,
    sample: str,
    niche_labels: pd.Series | None = None,
    knn=None,
    min_expr_frac: float = 0.02,
    min_cells_per_niche: int = 5,
) -> pd.DataFrame:
    """Infer lightweight spatial ligand-receptor communication edges.

    For each ligand-receptor pair, ligand exposure is computed by spatially lagging the
    ligand expression after masking to a sender niche. The receiver signal is receptor
    expression in the receiver niche. The reported LR score combines positive Spearman
    association, expression prevalence, and a simple spatial specificity term.
    """
    cols = [
        'sample', 'sender_niche', 'receiver_niche', 'ligand', 'receptor', 'lr_pair',
        'signature_id', 'signature_id_lr', 'signature_id_niche', 'n_sender_spots', 'n_receiver_spots', 'ligand_expr_frac', 'receptor_expr_frac',
        'ligand_exposure_mean', 'receptor_activity_mean', 'rho', 'pvalue', 'lr_score',
        'spatial_specificity', 'cascade_amplification', 'receptor_activation', 'exposure_strength'
    ]
    if expr.empty or coords.empty or lr_prior is None or lr_prior.empty:
        return pd.DataFrame(columns=cols)

    expr = expr.copy()
    expr.columns = expr.columns.astype(str).str.upper()
    lr = lr_prior.copy()
    lr['ligand'] = lr['ligand'].astype(str).str.upper()
    lr['receptor'] = lr['receptor'].astype(str).str.upper()
    lr = lr[lr['ligand'].isin(expr.columns) & lr['receptor'].isin(expr.columns)].drop_duplicates(['ligand', 'receptor'])
    if lr.empty:
        return pd.DataFrame(columns=cols)

    common = expr.index.intersection(coords.index)
    expr = expr.loc[common]
    coords = coords.loc[common]
    if niche_labels is None or len(niche_labels) == 0:
        niches = pd.Series('all', index=expr.index, name='niche')
    else:
        niches = niche_labels.reindex(expr.index).fillna('unknown').astype(str)
    if knn is None:
        knn = build_weighted_knn(coords, n_neighbors=min(12, max(1, len(coords) - 1)))

    # Precompute unmasked exposure for specificity background.
    ligands = sorted(set(lr['ligand']))
    background_lag = spatial_lag(expr[ligands], knn) if ligands else pd.DataFrame(index=expr.index)
    rows = []
    sender_values = sorted(niches.dropna().unique().tolist())
    receiver_values = sender_values
    for _, pair in lr.iterrows():
        ligand = str(pair['ligand'])
        receptor = str(pair['receptor'])
        lig_expr = expr[ligand].astype(float)
        rec_expr = expr[receptor].astype(float)
        lig_frac_global = float((lig_expr > 0).mean())
        rec_frac_global = float((rec_expr > 0).mean())
        if lig_frac_global < min_expr_frac or rec_frac_global < min_expr_frac:
            continue
        for sender in sender_values:
            sender_mask = (niches == sender).values
            if int(sender_mask.sum()) < min_cells_per_niche:
                continue
            masked = pd.DataFrame({ligand: lig_expr.values * sender_mask.astype(float)}, index=expr.index)
            sender_exposure = spatial_lag(masked, knn)[ligand].astype(float)
            for receiver in receiver_values:
                recv_mask = (niches == receiver)
                if int(recv_mask.sum()) < min_cells_per_niche:
                    continue
                exposure = sender_exposure.loc[recv_mask.index[recv_mask]].astype(float)
                receptor_values = rec_expr.loc[recv_mask.index[recv_mask]].astype(float)
                rho, pval = _safe_spearman(exposure, receptor_values)
                # Compare sender-masked exposure with the global ligand exposure in the same receiver niche.
                bg = background_lag.loc[recv_mask.index[recv_mask], ligand].astype(float) if ligand in background_lag.columns else exposure
                specificity = float(max(0.0, np.nanmean(_safe_zscore(exposure)) - np.nanmean(_safe_zscore(bg - exposure))))
                specificity = float(np.tanh(specificity)) if np.isfinite(specificity) else 0.0
                prevalence = np.sqrt(max(lig_frac_global, 0.0) * max(rec_frac_global, 0.0))
                # Cascade/amplification proxy: a ligand signal is stronger when neighborhood
                # exposure is both high and coherent with receptor abundance in the receiver niche.
                receptor_activation = float(np.tanh(max(np.nanmean(_safe_zscore(receptor_values)), 0.0))) if len(receptor_values) else 0.0
                exposure_strength = float(np.tanh(max(np.nanmean(_safe_zscore(exposure)), 0.0))) if len(exposure) else 0.0
                cascade_amplification = 1.0 + 0.25 * max(rho, 0.0) + 0.15 * exposure_strength + 0.10 * receptor_activation
                score = max(rho, 0.0) * prevalence * (0.5 + 0.5 * specificity) * cascade_amplification
                rows.append({
                    'sample': str(sample),
                    'sender_niche': str(sender),
                    'receiver_niche': str(receiver),
                    'ligand': ligand,
                    'receptor': receptor,
                    'lr_pair': f'{ligand}|{receptor}',
                    # Pair-level IDs are used for recurrent signatures; niche-specific IDs remain available for finer analysis.
                    'signature_id': _lr_pair_id(ligand, receptor),
                    'signature_id_lr': _lr_pair_id(ligand, receptor),
                    'signature_id_niche': _signature_id(ligand, receptor, str(sender), str(receiver)),
                    'n_sender_spots': int(sender_mask.sum()),
                    'n_receiver_spots': int(recv_mask.sum()),
                    'ligand_expr_frac': lig_frac_global,
                    'receptor_expr_frac': rec_frac_global,
                    'ligand_exposure_mean': float(np.nanmean(exposure)) if len(exposure) else 0.0,
                    'receptor_activity_mean': float(np.nanmean(receptor_values)) if len(receptor_values) else 0.0,
                    'rho': float(rho),
                    'pvalue': float(pval),
                    'lr_score': float(score),
                    'spatial_specificity': float(specificity),
                    'cascade_amplification': float(cascade_amplification),
                    'receptor_activation': float(receptor_activation),
                    'exposure_strength': float(exposure_strength),
                })
    if not rows:
        return pd.DataFrame(columns=cols)
    out = pd.DataFrame(rows)
    return out.sort_values(['lr_score', 'rho', 'spatial_specificity'], ascending=[False, False, False]).reset_index(drop=True)


def build_ccc_signatures(
    lr_edges: pd.DataFrame,
    min_sample_support: int = 2,
    min_score: float = 0.005,
    per_sample_quantile: float = 0.75,
) -> pd.DataFrame:
    """Extract recurrent CCC signatures from spatial LR edges.

    The previous implementation grouped by ligand, receptor, sender niche and receiver
    niche simultaneously. That was too strict for small Visium cohorts because the same
    LR pair can recur across samples while its niche labels shift. The primary signature
    unit is now ligand-receptor pair; dominant sender/receiver niches are retained as
    annotations, and niche-specific IDs remain in ``ccc_edges.csv`` for downstream plots.
    """
    cols = [
        'signature_id', 'ligand', 'receptor', 'dominant_sender_niche', 'dominant_receiver_niche',
        'n_sender_receiver_pairs', 'n_samples', 'n_crc_samples', 'n_nat_samples',
        'support_rate', 'crc_support_rate', 'nat_support_rate', 'crc_specificity', 'condition_label',
        'mean_lr_score', 'median_lr_score', 'mean_rho', 'min_pvalue', 'sample_list', 'niche_pairs'
    ]
    if lr_edges is None or lr_edges.empty:
        return pd.DataFrame(columns=cols)
    df = lr_edges.copy()
    if 'signature_id_lr' in df.columns:
        df['signature_id'] = df['signature_id_lr'].astype(str)
    else:
        df['signature_id'] = [_lr_pair_id(l, r) for l, r in zip(df['ligand'].astype(str), df['receptor'].astype(str))]
    df['lr_score'] = pd.to_numeric(df['lr_score'], errors='coerce').fillna(0.0)
    df['rho'] = pd.to_numeric(df.get('rho', 0.0), errors='coerce').fillna(0.0)
    df['pvalue'] = pd.to_numeric(df.get('pvalue', 1.0), errors='coerce').fillna(1.0)

    # Adaptive support: keep biologically plausible weak edges if they are among the
    # stronger spatial LR edges within their sample. This avoids zero signatures when
    # switching from a tiny seed prior to thousands of real-database LR pairs.
    if 'sample' in df.columns and df['sample'].nunique() > 0:
        q = df.groupby('sample')['lr_score'].transform(lambda s: s.quantile(float(per_sample_quantile)) if len(s) else 0.0)
    else:
        q = pd.Series(float(min_score), index=df.index)
    df['support_flag'] = (df['lr_score'] >= float(min_score)) | ((df['lr_score'] > 0) & (df['lr_score'] >= q))
    total_samples = max(1, df['sample'].nunique())
    rows = []
    for keys, sub in df.groupby(['signature_id', 'ligand', 'receptor']):
        supported = sub[sub['support_flag']].copy()
        n_support = supported['sample'].nunique()
        if n_support < int(min_sample_support):
            continue
        niche_pairs = (supported['sender_niche'].astype(str) + '->' + supported['receiver_niche'].astype(str)).value_counts()
        dominant_pair = niche_pairs.index[0] if len(niche_pairs) else 'unknown->unknown'
        dominant_sender, dominant_receiver = dominant_pair.split('->', 1) if '->' in dominant_pair else ('unknown', 'unknown')
        # MODIFIED: track condition recurrence separately from total recurrence. CRC-specific
        # support becomes a downstream ranking feature; shared signatures are retained but
        # no longer overinterpreted as tumor-specific communication.
        support_samples = sorted(supported['sample'].astype(str).unique().tolist())
        all_samples = sorted(df['sample'].astype(str).unique().tolist())
        crc_total = max(1, sum(_sample_condition(x) == 'CRC' for x in all_samples))
        nat_total = max(1, sum(_sample_condition(x) == 'NAT' for x in all_samples))
        n_crc_support = int(sum(_sample_condition(x) == 'CRC' for x in support_samples))
        n_nat_support = int(sum(_sample_condition(x) == 'NAT' for x in support_samples))
        crc_support_rate = float(n_crc_support / crc_total)
        nat_support_rate = float(n_nat_support / nat_total)
        crc_specificity = float(crc_support_rate - nat_support_rate)
        condition_label = 'CRC_enriched' if crc_specificity >= 0.34 else ('NAT_enriched' if crc_specificity <= -0.34 else 'shared')
        # END MODIFIED
        rows.append({
            'signature_id': keys[0],
            'ligand': keys[1],
            'receptor': keys[2],
            'dominant_sender_niche': dominant_sender,
            'dominant_receiver_niche': dominant_receiver,
            'n_sender_receiver_pairs': int(niche_pairs.shape[0]),
            'n_samples': int(n_support),
            'n_crc_samples': n_crc_support,
            'n_nat_samples': n_nat_support,
            'support_rate': float(n_support / total_samples),
            'crc_support_rate': crc_support_rate,
            'nat_support_rate': nat_support_rate,
            'crc_specificity': crc_specificity,
            'condition_label': condition_label,
            'mean_lr_score': float(supported['lr_score'].mean()),
            'median_lr_score': float(supported['lr_score'].median()),
            'mean_rho': float(supported['rho'].mean()),
            'min_pvalue': float(supported['pvalue'].min()),
            'sample_list': ';'.join(support_samples),
            'niche_pairs': ';'.join(niche_pairs.head(20).index.astype(str).tolist()),
        })
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows).sort_values(['support_rate', 'mean_lr_score', 'mean_rho'], ascending=[False, False, False]).reset_index(drop=True)


def _spot_sample(index_value: str) -> str:
    s = str(index_value)
    return s.split('|', 1)[0] if '|' in s else 'sample'


# MODIFIED: infer coarse sample condition from CRC/NAT sample IDs so recurrent CCC
# signatures can be marked as CRC-enriched, NAT-enriched, or shared. This avoids
# treating ubiquitous epithelial/background LR pairs as equally informative tumor
# microenvironment signals.
def _sample_condition(sample: object) -> str:
    text = str(sample).upper()
    if 'NAT' in text or 'NORMAL' in text or text.endswith('_N'):
        return 'NAT'
    return 'CRC'


def _within_sample_center(values: pd.Series, pseudos: list[str]) -> pd.Series:
    s = values.reindex(pseudos).astype(float).fillna(0.0)
    samples = pd.Series([p.split('|', 1)[0] for p in pseudos], index=pseudos)
    centered = s - s.groupby(samples).transform('mean')
    centered = centered.fillna(0.0)
    return centered
# END MODIFIED


def driver_niche_activity(spot_program_scores: pd.DataFrame, niche_labels: pd.Series) -> pd.DataFrame:
    cols = ['sample', 'receiver_niche', 'driver', 'driver_activity']
    if spot_program_scores is None or spot_program_scores.empty or niche_labels is None or niche_labels.empty:
        return pd.DataFrame(columns=cols)
    df = spot_program_scores.copy()
    df.index = df.index.astype(str)
    niches = niche_labels.reindex(df.index).fillna('unknown').astype(str)
    long = df.join(niches.rename('receiver_niche')).reset_index().rename(columns={'index': 'spot_id'})
    long['sample'] = long['spot_id'].map(_spot_sample)
    value_cols = [c for c in df.columns if c != 'receiver_niche']
    long = long.melt(id_vars=['spot_id', 'sample', 'receiver_niche'], value_vars=value_cols, var_name='driver', value_name='driver_activity')
    out = long.groupby(['sample', 'receiver_niche', 'driver'], as_index=False)['driver_activity'].mean()
    return out[cols]



def _deterministic_rng(*parts) -> np.random.Generator:
    seed_text = '|'.join(map(str, parts))
    seed = int(hashlib.sha1(seed_text.encode('utf-8')).hexdigest()[:8], 16)
    return np.random.default_rng(seed)


def _permutation_spearman_pvalue(x: np.ndarray, y: np.ndarray, n_perm: int = 999, seed_parts: tuple = ()) -> tuple[float, float]:
    """One-sided permutation p-value for positive Spearman association."""
    rho, _ = _safe_spearman(x, y)
    if len(x) < 4 or rho <= 0:
        return float(rho), 1.0
    rng = _deterministic_rng(*seed_parts)
    ge = 1
    valid = 1
    y = np.asarray(y, dtype=float)
    for _ in range(int(n_perm)):
        yp = rng.permutation(y)
        rp, _ = _safe_spearman(x, yp)
        if np.isfinite(rp):
            valid += 1
            if rp >= rho:
                ge += 1
    return float(rho), float(ge / max(valid, 1))


def _bootstrap_positive_stability(x: np.ndarray, y: np.ndarray, n_boot: int = 200, seed_parts: tuple = ()) -> float:
    """Fraction of bootstrap resamples preserving positive association."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if n < 4 or np.nanstd(x) <= 1e-12 or np.nanstd(y) <= 1e-12:
        return 0.0
    rng = _deterministic_rng(*seed_parts)
    ok = 0
    total = 0
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(idx)) < 3:
            continue
        r, _ = _safe_spearman(x[idx], y[idx])
        if np.isfinite(r):
            total += 1
            ok += int(r > 0)
    return float(ok / max(total, 1))


def associate_driver_ccc(
    spot_program_scores: pd.DataFrame,
    niche_labels: pd.Series,
    lr_edges: pd.DataFrame,
    ccc_signatures: pd.DataFrame | None = None,
    min_observations: int = 3,
    driver_quantile: float = 0.70,
    ccc_quantile: float = 0.70,
    n_permutations: int = 499,
    n_bootstrap: int = 200,
) -> pd.DataFrame:
    """Robustly associate driver programs with CCC signatures.

    The old implementation used a one-sided Fisher test on sample×niche pseudo-samples
    as the main inferential quantity. That inflated the apparent sample size when only a
    few real patients were available. The revised implementation keeps pseudo-samples as
    a bootstrap/stability layer, but uses real-sample driver-program activity versus CCC
    signature activity as the primary association test.

    Reported quantities:
    * sample_rho/sample_pvalue: sample-level Spearman + permutation test.
    * pseudo_fisher_pvalue: pseudo-sample co-presence support, used only as evidence.
    * bootstrap_stability: fraction of pseudo-sample bootstraps with positive rho.
    * assoc_score: transparent composite score for ranking candidate IE links.
    * significant: strict exploratory gate (q<0.1, support/stability positive).
    """
    cols = [
        'driver', 'signature_id', 'ligand', 'receptor', 'sender_niche', 'receiver_niche',
        'assoc_score', 'rho', 'sample_rho', 'pseudo_spearman_rho', 'niche_residual_rho', 'odds_ratio', 'pvalue',
        'sample_pvalue', 'niche_residual_pvalue', 'pseudo_fisher_pvalue', 'driver2comm_fisher_pvalue',
        'combined_pvalue', 'qvalue', 'sample_qvalue', 'niche_qvalue', 'fisher_qvalue', 'significant', 'direction',
        'n_observations', 'n_samples', 'n_driver_present', 'n_signature_present',
        'n_copresent', 'copresent_rate', 'mean_driver_activity', 'mean_lr_score',
        'support_rate', 'crc_specificity', 'condition_label', 'bootstrap_stability', 'test'
    ]
    if spot_program_scores is None or spot_program_scores.empty or lr_edges is None or lr_edges.empty:
        return pd.DataFrame(columns=cols)
    activity = driver_niche_activity(spot_program_scores, niche_labels)
    if activity.empty:
        return pd.DataFrame(columns=cols)
    df = lr_edges.copy()
    # MODIFIED: preserve signature-level CRC/NAT specificity metadata for association
    # ranking and reporting.
    sig_meta = {}
    if ccc_signatures is not None and not ccc_signatures.empty:
        keep = set(ccc_signatures['signature_id'].astype(str))
        df = df[df['signature_id'].astype(str).isin(keep)].copy()
        meta_cols = [c for c in ['signature_id', 'support_rate', 'crc_specificity', 'condition_label'] if c in ccc_signatures.columns]
        if meta_cols:
            sig_meta = ccc_signatures[meta_cols].drop_duplicates('signature_id').set_index('signature_id').to_dict('index')
    # END MODIFIED
    if df.empty:
        return pd.DataFrame(columns=cols)
    df['sample'] = df['sample'].astype(str)
    df['receiver_niche'] = df['receiver_niche'].astype(str)
    df['pseudo_sample'] = df['sample'] + '|' + df['receiver_niche']
    activity = activity.copy()
    activity['sample'] = activity['sample'].astype(str)
    activity['receiver_niche'] = activity['receiver_niche'].astype(str)
    activity['pseudo_sample'] = activity['sample'] + '|' + activity['receiver_niche']

    # Signature activity at pseudo-sample and real-sample levels.
    sig_rows = []
    for sig, sub in df.groupby('signature_id'):
        vals = pd.to_numeric(sub['lr_score'], errors='coerce').fillna(0.0)
        thr = float(max(vals.quantile(float(ccc_quantile)), vals.mean() + 0.25 * vals.std(ddof=0), 1e-12))
        agg = sub.assign(_present=(vals >= thr) & (vals > 0)).groupby('pseudo_sample', as_index=False).agg(
            ccc_present=('_present', 'max'),
            lr_score=('lr_score', 'mean'),
            ligand=('ligand', 'first'),
            receptor=('receptor', 'first'),
            sender_niche=('sender_niche', lambda x: x.astype(str).value_counts().index[0]),
            receiver_niche=('receiver_niche', 'first'),
            sample=('sample', 'first'),
        )
        agg['signature_id'] = str(sig)
        sig_rows.append(agg)
    sig_presence = pd.concat(sig_rows, ignore_index=True) if sig_rows else pd.DataFrame()
    if sig_presence.empty:
        return pd.DataFrame(columns=cols)

    all_pseudos = sorted(set(activity['pseudo_sample']).union(sig_presence['pseudo_sample']))
    all_samples = sorted(set(activity['sample']).union(sig_presence['sample']))
    rows = []
    for driver, act_sub in activity.groupby('driver'):
        # Pseudo-sample driver activity for stability layer.
        act_pseudo = act_sub.groupby('pseudo_sample')['driver_activity'].mean().reindex(all_pseudos).fillna(0.0)
        if act_pseudo.nunique(dropna=True) <= 1:
            driver_present = act_pseudo > 0
        else:
            driver_present = act_pseudo >= float(act_pseudo.quantile(float(driver_quantile)))
        # Real-sample driver activity is the primary unit of inference.
        act_sample = act_sub.groupby('sample')['driver_activity'].mean().reindex(all_samples).fillna(0.0)
        for sig, sig_sub in sig_presence.groupby('signature_id'):
            sig_present = sig_sub.groupby('pseudo_sample')['ccc_present'].max().reindex(all_pseudos, fill_value=False).astype(bool)
            lr_pseudo = sig_sub.groupby('pseudo_sample')['lr_score'].mean().reindex(all_pseudos).fillna(0.0)
            lr_sample = sig_sub.groupby('sample')['lr_score'].mean().reindex(all_samples).fillna(0.0)
            n_obs = int(len(all_pseudos))
            n_samples = int(len(all_samples))
            if n_obs < int(min_observations) or n_samples < 2:
                continue
            a = int((driver_present & sig_present).sum())
            b = int((driver_present & ~sig_present).sum())
            c = int((~driver_present & sig_present).sum())
            d = int((~driver_present & ~sig_present).sum())
            try:
                odds, fisher_p = fisher_exact([[a, b], [c, d]], alternative='greater')
            except Exception:
                odds, fisher_p = 0.0, 1.0
            pseudo_rho, _ = _safe_spearman(act_pseudo.values, lr_pseudo.values)
            # MODIFIED: Keep the Driver2Comm one-hot Fisher test, but add a within-sample
            # centered pseudo-sample association. This removes global patient/sample effects
            # that caused real_run(1) to produce rho=1 for many generic CCC signatures.
            sample_rho, sample_p = _permutation_spearman_pvalue(
                act_sample.values, lr_sample.values,
                n_perm=n_permutations if n_samples >= 4 else 0,
                seed_parts=(driver, sig, 'sample_perm'),
            )
            if n_samples < 4:
                sample_p = 1.0
            act_resid = _within_sample_center(act_pseudo, all_pseudos)
            lr_resid = _within_sample_center(lr_pseudo, all_pseudos)
            niche_rho, niche_p = _permutation_spearman_pvalue(
                act_resid.values, lr_resid.values,
                n_perm=n_permutations if n_obs >= 12 else max(99, int(n_permutations / 2)),
                seed_parts=(driver, sig, 'niche_residual_perm'),
            )
            try:
                combined_p = float(combine_pvalues([
                    max(min(float(sample_p), 1.0), 1e-12),
                    max(min(float(niche_p), 1.0), 1e-12),
                    max(min(float(fisher_p), 1.0), 1e-12),
                ], method='fisher')[1])
            except Exception:
                combined_p = max(float(sample_p), float(niche_p), float(fisher_p))
            # END MODIFIED
            stability = _bootstrap_positive_stability(
                act_pseudo.values, lr_pseudo.values,
                n_boot=n_bootstrap,
                seed_parts=(driver, sig, 'pseudo_boot'),
            )
            first = sig_sub.iloc[0]
            meta = sig_meta.get(str(sig), {}) if isinstance(sig_meta, dict) else {}
            support_rate = float(meta.get('support_rate', sig_sub['sample'].nunique() / max(1, df['sample'].nunique())))
            crc_specificity = float(meta.get('crc_specificity', 0.0))
            condition_label = str(meta.get('condition_label', 'unknown'))
            copresent_rate = float(a / max(1, n_obs))
            odds_term = float(np.tanh(np.log1p(odds if np.isfinite(odds) else (n_obs + 1.0)) / 3.0))
            # Replication-oriented rank score. It does not pretend to be a p-value.
            assoc_score = float(
                0.25 * max(niche_rho, 0.0)
                + 0.20 * max(sample_rho, 0.0)
                + 0.18 * stability
                + 0.15 * support_rate
                + 0.10 * copresent_rate
                + 0.07 * odds_term
                + 0.05 * max(crc_specificity, 0.0)
            )
            rows.append({
                'driver': str(driver),
                'signature_id': str(sig),
                'ligand': str(first['ligand']),
                'receptor': str(first['receptor']),
                'sender_niche': str(first['sender_niche']),
                'receiver_niche': str(first['receiver_niche']),
                'assoc_score': assoc_score,
                'rho': float(sample_rho),
                'sample_rho': float(sample_rho),
                'pseudo_spearman_rho': float(pseudo_rho),
                'niche_residual_rho': float(niche_rho),
                'odds_ratio': float(odds) if np.isfinite(odds) else float(n_obs + 1),
                # MODIFIED: expose the paper-aligned Fisher p-value and the combined p-value.
                'pvalue': float(combined_p),
                'sample_pvalue': float(sample_p),
                'niche_residual_pvalue': float(niche_p),
                'pseudo_fisher_pvalue': float(fisher_p),
                'driver2comm_fisher_pvalue': float(fisher_p),
                'combined_pvalue': float(combined_p),
                # END MODIFIED
                'direction': 'positive' if sample_rho > 0 else ('negative' if sample_rho < 0 else 'flat'),
                'n_observations': n_obs,
                'n_samples': n_samples,
                'n_driver_present': int(driver_present.sum()),
                'n_signature_present': int(sig_present.sum()),
                'n_copresent': a,
                'copresent_rate': copresent_rate,
                'mean_driver_activity': float(act_sample.mean()),
                'mean_lr_score': float(lr_sample.mean()),
                'support_rate': support_rate,
                'crc_specificity': crc_specificity,
                'condition_label': condition_label,
                'bootstrap_stability': stability,
                'test': 'sample_level_spearman_permutation_plus_pseudosample_bootstrap',
            })
    if not rows:
        return pd.DataFrame(columns=cols)
    out = pd.DataFrame(rows)
    # MODIFIED: separate FDR columns for the combined test, sample-level test, and
    # Driver2Comm pseudo-sample Fisher test. The significance gate requires both the
    # combined association and positive/stable pseudo-sample support.
    out['qvalue'] = _bh_fdr(out['pvalue'])
    out['sample_qvalue'] = _bh_fdr(out['sample_pvalue'])
    out['niche_qvalue'] = _bh_fdr(out['niche_residual_pvalue'])
    out['fisher_qvalue'] = _bh_fdr(out['pseudo_fisher_pvalue'])
    out['significant'] = (
        (out['qvalue'].fillna(1.0) < 0.10)
        & (out['sample_rho'].fillna(0.0) > 0)
        & (out['niche_residual_rho'].fillna(0.0) > 0.10)
        & (out['niche_qvalue'].fillna(1.0) <= 0.20)
        & (out['pseudo_fisher_pvalue'].fillna(1.0) <= 0.10)
        & (out['support_rate'].fillna(0.0) >= 0.60)
        & (out['crc_specificity'].fillna(0.0) >= -0.10)
        & (out['bootstrap_stability'].fillna(0.0) >= 0.70)
    )
    # END MODIFIED
    return out.sort_values(['significant', 'qvalue', 'assoc_score', 'bootstrap_stability'], ascending=[False, True, False, False]).reset_index(drop=True)[cols]

def build_ie_pathways(
    driver_programs: pd.DataFrame,
    driver_ccc_associations: pd.DataFrame,
    max_targets_per_driver: int = 5,
) -> pd.DataFrame:
    """Materialize interpretable intrinsic-extrinsic paths.

    Driver2Comm defines IE pathways as shortest paths connecting an intrinsic driver to
    its associated CCC signature in the expanded intracellular/intercellular network. In
    this lightweight package we expand the graph with the inferred GRN program edges and
    LR edges, then compute shortest paths when possible; if the graph is disconnected we
    retain an explicit driver→program→CCC fallback row rather than silently dropping the
    biology.
    """
    cols = [
        'driver', 'target_program_gene', 'ligand', 'receptor', 'signature_id', 'sender_niche',
        'receiver_niche', 'path_type', 'path_length', 'path_nodes', 'internal_score', 'external_score',
        'ccc_score', 'association_qvalue', 'association_direction'
    ]
    if driver_programs is None or driver_programs.empty or driver_ccc_associations is None or driver_ccc_associations.empty:
        return pd.DataFrame(columns=cols)
    programs = driver_programs.copy()
    try:
        import networkx as nx
        graph = nx.DiGraph()
        for _, r in programs.iterrows():
            graph.add_edge(str(r['driver']), str(r['target']), kind='grn', weight=float(r.get('evidence_score', r.get('score', 0.0))))
        for _, a in driver_ccc_associations.iterrows():
            lig = str(a['ligand'])
            rec = str(a['receptor'])
            sig = str(a['signature_id'])
            graph.add_edge(lig, rec, kind='lr', weight=float(a.get('assoc_score', 0.0)))
            graph.add_edge(rec, sig, kind='signature', weight=float(a.get('assoc_score', 0.0)))
    except Exception:
        nx = None
        graph = None

    rows = []
    for driver, assoc in driver_ccc_associations.groupby('driver'):
        # MODIFIED: Prefer statistically supported driver-CCC links for IE paths when available.
        if 'significant' in assoc.columns and assoc['significant'].astype(bool).any():
            assoc = assoc[assoc['significant'].astype(bool)].copy()
        # END MODIFIED
        targets = programs[programs['driver'].astype(str) == str(driver)].copy()
        if targets.empty:
            continue
        sort_cols = [c for c in ['evidence_score', 'score', 'internal_support'] if c in targets.columns]
        if sort_cols:
            targets = targets.sort_values(sort_cols, ascending=[False] * len(sort_cols))
        targets = targets.head(max_targets_per_driver)
        for _, a in assoc.head(20).iterrows():
            lig = str(a['ligand'])
            rec = str(a['receptor'])
            sig = str(a['signature_id'])
            shortest = None
            path_type = 'driver_grn_to_spatial_ccc'
            if graph is not None:
                for sink in (lig, rec, sig):
                    try:
                        path = nx.shortest_path(graph, source=str(driver), target=sink)
                        if shortest is None or len(path) < len(shortest):
                            shortest = path
                    except Exception:
                        pass
            for _, t in targets.iterrows():
                target_gene = str(t['target'])
                if shortest is not None:
                    # MODIFIED: Avoid duplicating a shortest path under unrelated program
                    # targets. The reported target_program_gene must be part of the path.
                    if target_gene not in set(map(str, shortest[1:-1])):
                        continue
                    path_nodes = shortest if shortest[-1] == sig else shortest + ([sig] if shortest[-1] != sig else [])
                    path_length = max(1, len(path_nodes) - 1)
                    local_type = 'shortest_expanded_ie_path'
                    # END MODIFIED
                else:
                    # MODIFIED: Do not fabricate a mechanistic target-to-ligand edge when the
                    # target is not part of the LR signature. Non-direct bridges are retained
                    # for review but explicitly labelled as unverified evidence bridges.
                    direct = target_gene.upper() in {lig.upper(), rec.upper()}
                    if direct:
                        path_nodes = [str(driver), target_gene, sig]
                        path_length = 2
                        local_type = 'direct_target_to_ccc_gene'
                    else:
                        path_nodes = [str(driver), target_gene, 'UNVERIFIED_BRIDGE', lig, rec, sig]
                        path_length = 5
                        local_type = 'candidate_unverified_bridge'
                    # END MODIFIED
                rows.append({
                    'driver': str(driver),
                    'target_program_gene': target_gene,
                    'ligand': lig,
                    'receptor': rec,
                    'signature_id': sig,
                    'sender_niche': str(a['sender_niche']),
                    'receiver_niche': str(a['receiver_niche']),
                    'path_type': local_type,
                    'path_length': int(path_length),
                    'path_nodes': '->'.join(map(str, path_nodes)),
                    'internal_score': float(t.get('internal_support', t.get('internal', 0.0))),
                    'external_score': float(t.get('external_support', t.get('external', 0.0))),
                    'ccc_score': float(a.get('assoc_score', 0.0)),
                    'association_qvalue': float(a.get('qvalue', 1.0)),
                    'association_direction': str(a.get('direction', 'unknown')),
                })
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows).sort_values(['association_qvalue', 'ccc_score', 'path_length', 'internal_score'], ascending=[True, False, True, False]).reset_index(drop=True)[cols]
