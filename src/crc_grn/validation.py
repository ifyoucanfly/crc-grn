from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from .io_matrix import read_mex_folder, list_mex_sources, list_xenium_runs, read_xenium_run, read_xenium_gene_panel_json, _find_first, _archive_sample_name


def _program_raw_score(expr: pd.DataFrame, program_df) -> pd.Series:
    if isinstance(program_df, pd.DataFrame):
        targets = program_df['target'].astype(str).tolist()
        weight_col = 'evidence_score' if 'evidence_score' in program_df.columns else ('score' if 'score' in program_df.columns else None)
        if weight_col is not None:
            weights = program_df.set_index('target')[weight_col].astype(float)
        else:
            weights = pd.Series(1.0, index=targets)
    else:
        targets = list(map(str, program_df))
        weights = pd.Series(1.0, index=targets)
    genes = [g for g in targets if g in expr.columns]
    if not genes:
        return pd.Series(0.0, index=expr.index, dtype=float)
    sub = expr[genes].copy().astype(float)
    std = sub.std(axis=0).replace(0, np.nan)
    sub = ((sub - sub.mean(axis=0)) / std).fillna(0.0)
    w = weights.reindex(genes).fillna(0.0).clip(lower=0.0)
    if float(w.sum()) <= 0:
        w = pd.Series(1.0, index=genes)
    raw = sub.mul(w.values, axis=1).sum(axis=1) / float(w.sum())
    raw = raw.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return raw.astype(float)


def _safe_program_score(expr: pd.DataFrame, program_df) -> pd.Series:
    raw = _program_raw_score(expr, program_df)
    if raw.nunique(dropna=True) <= 1:
        return pd.Series(0.5, index=expr.index, dtype=float)
    ranked = raw.rank(method='average', pct=True)
    return ranked.astype(float)


def _program_activity_summary(expr: pd.DataFrame, program_df) -> dict:
    raw = _program_raw_score(expr, program_df)
    if raw.empty:
        return {'activity': 0.0, 'mean_raw': 0.0, 'top10_raw': 0.0, 'iqr_raw': 0.0}
    n = len(raw)
    k = max(10, int(np.ceil(0.10 * n))) if n >= 10 else max(1, int(np.ceil(0.20 * n)))
    top_mean = float(raw.nlargest(min(k, n)).mean()) if n else 0.0
    mean_raw = float(raw.mean()) if n else 0.0
    iqr_raw = float(raw.quantile(0.75) - raw.quantile(0.25)) if n else 0.0
    # emphasize focal activation while still keeping comparability across samples
    activity = float(top_mean)
    return {'activity': activity, 'mean_raw': mean_raw, 'top10_raw': top_mean, 'iqr_raw': iqr_raw}




def _coalesce_score_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for base in ['internal', 'external', 'internal_support', 'external_support', 'full_r2', 'corr_xy', 'corr_yz', 'corr_xz']:
        if base in df.columns:
            continue
        alts = [f'{base}_x', f'{base}_y', f'{base}_mean', f'{base}_pooled']
        found = [c for c in alts if c in df.columns]
        if found:
            s = None
            for c in found:
                if s is None:
                    s = df[c]
                else:
                    s = s.combine_first(df[c])
            df[base] = s
    return df

def build_driver_programs(
    edge_scores: pd.DataFrame,
    top_targets_per_driver: int = 15,
    min_score: float = 0.0,
    min_qvalue: float = 0.25,
    min_edges_per_driver: int = 3,
    min_sample_support: int = 3,
    min_consistency: float = 0.60,
) -> pd.DataFrame:
    if edge_scores.empty:
        return pd.DataFrame(columns=['driver', 'target', 'score', 'internal', 'external'])
    df = _coalesce_score_columns(edge_scores)
    df = df[df['score'] >= min_score].copy()
    if 'qvalue' in df.columns:
        df = df[df['qvalue'].fillna(1.0) <= min_qvalue].copy()
    if 'n_samples' in df.columns:
        df = df[df['n_samples'].fillna(0).astype(int) >= min_sample_support].copy()
    if 'consistency_pos' in df.columns:
        df = df[df['consistency_pos'].fillna(0.0) >= min_consistency].copy()
    df = df[df['source'].astype(str) != df['target'].astype(str)].copy()
    sort_cols = [c for c in ['source', 'evidence_score', 'score', 'external_support', 'internal_support', 'internal'] if c in df.columns]
    ascending = [True] + [False] * (len(sort_cols) - 1)
    df = df.sort_values(sort_cols, ascending=ascending)
    df['driver'] = df['source']
    out_cols = [c for c in ['driver', 'target', 'score', 'evidence_score', 'internal', 'external', 'internal_support', 'external_support', 'qvalue', 'n_samples', 'consistency_pos'] if c in df.columns]
    out = df.groupby('driver').head(top_targets_per_driver)[out_cols].reset_index(drop=True)
    if out.empty:
        return out
    keep_drivers = out.groupby('driver')['target'].count()
    keep_drivers = keep_drivers[keep_drivers >= int(min_edges_per_driver)].index.astype(str)
    out = out[out['driver'].astype(str).isin(set(keep_drivers))].reset_index(drop=True)
    return out


def summarize_driver_ranking(edge_scores: pd.DataFrame, sample_edge_scores: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    if edge_scores.empty:
        return pd.DataFrame(columns=['driver', 'n_edges', 'mean_score', 'mean_internal', 'mean_external'])
    edge_scores = _coalesce_score_columns(edge_scores)
    required = {'source', 'target', 'score'}
    missing = required.difference(edge_scores.columns)
    if missing:
        raise KeyError(f'edge_scores is missing required columns for driver ranking: {sorted(missing)}')

    agg_map = {
        'n_edges': ('target', 'count'),
        'mean_score': ('score', 'mean'),
        'max_score': ('score', 'max'),
    }
    if 'internal' in edge_scores.columns:
        agg_map['mean_internal'] = ('internal', 'mean')
    if 'external' in edge_scores.columns:
        agg_map['mean_external'] = ('external', 'mean')
    if 'internal_support' in edge_scores.columns:
        agg_map['mean_internal_support'] = ('internal_support', 'mean')
    if 'external_support' in edge_scores.columns:
        agg_map['mean_external_support'] = ('external_support', 'mean')

    if 'qvalue' in edge_scores.columns:
        sig_mask = edge_scores['qvalue'].fillna(1.0) <= 0.25
        sig_counts = edge_scores.assign(sig=sig_mask.astype(int)).groupby('source')['sig'].sum().reset_index().rename(columns={'source': 'driver', 'sig': 'n_significant_edges'})
    else:
        sig_counts = None

    grp = edge_scores.groupby('source').agg(**agg_map).reset_index().rename(columns={'source': 'driver'})
    if 'mean_internal' not in grp.columns:
        grp['mean_internal'] = np.nan
    if 'mean_external' not in grp.columns:
        grp['mean_external'] = np.nan
    if sig_counts is not None:
        grp = grp.merge(sig_counts, on='driver', how='left')

    external_support = grp['mean_external_support'].fillna(0.0) if 'mean_external_support' in grp.columns else pd.Series(0.0, index=grp.index)
    grp['evidence_score'] = (
        grp['mean_score'].fillna(0.0)
        * np.log1p(grp['n_edges'].fillna(0.0))
        * (1.0 + external_support)
    )
    grp = grp.sort_values(['evidence_score', 'mean_score', 'max_score', 'n_edges'], ascending=[False, False, False, False]).reset_index(drop=True)
    grp['global_rank'] = np.arange(1, len(grp) + 1)

    if sample_edge_scores is not None and not sample_edge_scores.empty and {'sample', 'source', 'score'}.issubset(sample_edge_scores.columns):
        sample_rank = sample_edge_scores.groupby(['sample', 'source'])['score'].mean().reset_index()
        sample_rank['sample_rank'] = sample_rank.groupby('sample')['score'].rank(ascending=False, method='average')
        stab = sample_rank.groupby('source').agg(
            n_samples=('sample', 'nunique'),
            mean_sample_rank=('sample_rank', 'mean'),
            std_sample_rank=('sample_rank', 'std'),
        ).reset_index().rename(columns={'source': 'driver'})
        stab['stability_score'] = 1.0 / (1.0 + stab['std_sample_rank'].fillna(0.0))
        grp = grp.merge(stab, on='driver', how='left')
        grp['evidence_score'] = grp['evidence_score'] * (1.0 + 0.5 * grp['stability_score'].fillna(0.0))
        grp = grp.sort_values(['evidence_score', 'mean_score', 'max_score', 'n_edges'], ascending=[False, False, False, False]).reset_index(drop=True)
        grp['global_rank'] = np.arange(1, len(grp) + 1)
    else:
        grp['n_samples'] = grp.get('n_samples', np.nan)
        grp['mean_sample_rank'] = grp.get('mean_sample_rank', np.nan)
        grp['std_sample_rank'] = grp.get('std_sample_rank', np.nan)
        grp['stability_score'] = grp.get('stability_score', np.nan)
    return grp


def derive_niches(coords: pd.DataFrame, n_niches: int = 6) -> pd.Series:
    n = len(coords)
    if n == 0:
        return pd.Series(dtype=str)
    # MODIFIED: derive niches within each sample when spot IDs contain a sample prefix.
    # Pooling coordinates across unrelated patients caused artificial niche labels and
    # weakened the downstream cell-wise / niche-wise validation requested in the meeting.
    index_as_str = pd.Index(coords.index.astype(str))
    if any('|' in x for x in index_as_str):
        sample_labels = pd.Series([x.split('|', 1)[0] if '|' in x else 'sample' for x in index_as_str], index=coords.index)
        parts = []
        for sample, idx in sample_labels.groupby(sample_labels).groups.items():
            sub = coords.loc[list(idx)]
            ns = len(sub)
            if ns == 0:
                continue
            k = max(2, min(n_niches, ns))
            arr = StandardScaler().fit_transform(sub[['x', 'y']].values)
            if ns <= k:
                labels = np.arange(ns)
            else:
                km = KMeans(n_clusters=k, n_init=10, random_state=42)
                labels = km.fit_predict(arr)
            parts.append(pd.Series([f'{sample}:niche_{int(i)}' for i in labels], index=sub.index, name='niche'))
        return pd.concat(parts).reindex(coords.index).rename('niche') if parts else pd.Series(dtype=str)
    # END MODIFIED
    k = max(2, min(n_niches, n))
    arr = StandardScaler().fit_transform(coords[['x', 'y']].values)
    if n <= k:
        labels = np.arange(n)
    else:
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(arr)
    return pd.Series([f'niche_{int(i)}' for i in labels], index=coords.index, name='niche')


def score_programs(expr: pd.DataFrame, coords: pd.DataFrame, driver_programs: pd.DataFrame, n_niches: int = 6):
    if driver_programs.empty:
        empty = pd.DataFrame(index=expr.index)
        return empty, pd.DataFrame(), pd.Series(dtype=str)
    spot_scores = {}
    for driver, df in driver_programs.groupby('driver'):
        spot_scores[str(driver)] = _safe_program_score(expr, df)
    spot_scores = pd.DataFrame(spot_scores, index=expr.index)
    niche_labels = derive_niches(coords, n_niches=n_niches)
    niche_scores = spot_scores.join(niche_labels, how='left').groupby('niche').mean(numeric_only=True)
    return spot_scores, niche_scores, niche_labels


def summarize_niche_specificity(spot_program_scores: pd.DataFrame, niche_labels: pd.Series) -> pd.DataFrame:
    if spot_program_scores.empty or niche_labels.empty:
        return pd.DataFrame(columns=['driver', 'best_niche', 'best_niche_score', 'second_best_niche_score', 'specificity_margin'])
    joined = spot_program_scores.join(niche_labels.rename('niche'), how='left')
    niche_means = joined.groupby('niche').mean(numeric_only=True).T
    rows = []
    for driver, vals in niche_means.iterrows():
        vals = vals.sort_values(ascending=False)
        best = float(vals.iloc[0]) if len(vals) >= 1 else np.nan
        second = float(vals.iloc[1]) if len(vals) >= 2 else 0.0
        rows.append({
            'driver': str(driver),
            'best_niche': str(vals.index[0]) if len(vals) else None,
            'best_niche_score': best,
            'second_best_niche_score': second,
            'specificity_margin': float(best - second),
        })
    return pd.DataFrame(rows).sort_values(['specificity_margin', 'best_niche_score'], ascending=[False, False]).reset_index(drop=True)


def external_replication(discovery_driver_ranking: pd.DataFrame, driver_programs: pd.DataFrame, external_root: str | Path, genes: Optional[Iterable[str]] = None) -> dict:
    external_root = Path(external_root)
    sample_sources = list_mex_sources(external_root)
    if not external_root.exists() or not sample_sources:
        return {'available': False, 'sample_summaries': pd.DataFrame(), 'driver_replication': pd.DataFrame(), 'program_activity': pd.DataFrame()}

    if genes is None:
        requested_gene_iter = []
    elif isinstance(genes, pd.Index):
        requested_gene_iter = genes.astype(str).tolist()
    else:
        requested_gene_iter = [str(g) for g in genes]

    requested_genes = set(requested_gene_iter) | set(driver_programs['target'].astype(str).tolist()) | set(discovery_driver_ranking['driver'].astype(str).tolist())
    sample_rows = []
    activity_rows = []
    driver_activity_rows = []
    for sample_src in sample_sources:
        sample_name = sample_src.name if sample_src.is_dir() else _archive_sample_name(sample_src)
        try:
            expr, _coords = read_mex_folder(sample_src, genes=requested_genes, prefix=f'{sample_name}|', require_positions=False)
        except Exception as e:
            sample_rows.append({'sample': sample_name, 'source_type': 'archive' if sample_src.is_file() else 'directory', 'status': 'failed', 'error': str(e), 'n_spots': 0, 'n_measured_genes': 0})
            continue
        if expr.empty:
            sample_rows.append({'sample': sample_name, 'source_type': 'archive' if sample_src.is_file() else 'directory', 'status': 'empty', 'error': '', 'n_spots': 0, 'n_measured_genes': 0})
            continue
        for driver, df in driver_programs.groupby('driver'):
            summ = _program_activity_summary(expr, df)
            activity_rows.append({
                'sample': sample_name,
                'driver': driver,
                'program_activity': float(summ['activity']),
                'program_activity_mean_raw': float(summ['mean_raw']),
                'program_activity_top10_raw': float(summ['top10_raw']),
                'program_activity_iqr_raw': float(summ['iqr_raw']),
                'n_targets_measured': int(sum(t in expr.columns for t in df['target'])),
                    'n_targets_total': int(df['target'].astype(str).nunique()),
                    'target_coverage_rate': float(int(sum(t in expr.columns for t in df['target'])) / max(1, int(df['target'].astype(str).nunique()))),
                })
        driver_means = {d: float(expr[d].mean()) for d in discovery_driver_ranking['driver'].astype(str) if d in expr.columns}
        for d, val in driver_means.items():
            driver_activity_rows.append({'sample': sample_name, 'driver': d, 'driver_expression': val})
        sample_rows.append({'sample': sample_name, 'source_type': 'archive' if sample_src.is_file() else 'directory', 'status': 'loaded', 'error': '', 'n_spots': int(expr.shape[0]), 'n_measured_genes': int(expr.shape[1])})

    program_activity = pd.DataFrame(activity_rows)
    driver_expr = pd.DataFrame(driver_activity_rows)
    if program_activity.empty:
        return {'available': False, 'sample_summaries': pd.DataFrame(sample_rows), 'driver_replication': pd.DataFrame(), 'program_activity': pd.DataFrame(), 'driver_expression': pd.DataFrame()}

    # within-sample normalization prevents all drivers collapsing toward the same global midpoint
    for raw_col in ['program_activity', 'program_activity_mean_raw', 'program_activity_top10_raw']:
        z_col = f'{raw_col}_z'
        rk_col = f'{raw_col}_rankpct'
        program_activity[z_col] = program_activity.groupby('sample')[raw_col].transform(lambda s: ((s - s.mean()) / (s.std(ddof=0) if float(s.std(ddof=0)) > 1e-12 else 1.0)).astype(float))
        program_activity[rk_col] = program_activity.groupby('sample')[raw_col].rank(ascending=False, pct=True, method='average').astype(float)

    ext_rank = program_activity.groupby('driver').agg(
        program_activity=('program_activity', 'mean'),
        program_activity_std=('program_activity', 'std'),
        program_activity_mean_raw=('program_activity_mean_raw', 'mean'),
        program_activity_top10_raw=('program_activity_top10_raw', 'mean'),
        program_activity_iqr_raw=('program_activity_iqr_raw', 'mean'),
        mean_activity_z=('program_activity_top10_raw_z', 'mean'),
        mean_activity_rankpct=('program_activity_top10_raw_rankpct', 'mean'),
        n_ext_samples=('sample', 'nunique'),
        mean_targets_measured=('n_targets_measured', 'mean'),
        mean_target_coverage_rate=('target_coverage_rate', 'mean'),
    ).reset_index()

    # agreement-oriented ranking: prioritize drivers consistently elevated across samples,
    # especially among the strongest focal program activations.
    z = ext_rank['mean_activity_z'].fillna(0.0)
    top = ext_rank['program_activity_top10_raw'].fillna(0.0)
    top_scaled = (top - top.mean()) / (top.std(ddof=0) if float(top.std(ddof=0)) > 1e-12 else 1.0)
    ext_rank['coverage_pass'] = (ext_rank['mean_targets_measured'].fillna(0) >= 2) | (ext_rank['mean_target_coverage_rate'].fillna(0) >= 0.30)
    ext_rank['agreement_score_raw'] = 0.60 * z + 0.15 * ext_rank['mean_activity_rankpct'].fillna(0.0) + 0.25 * top_scaled
    ext_rank['agreement_score'] = ext_rank['agreement_score_raw'] * np.where(ext_rank['coverage_pass'], 1.0, 0.35)
    ext_rank = ext_rank.sort_values(['agreement_score', 'mean_activity_rankpct', 'program_activity_top10_raw'], ascending=False).reset_index(drop=True)
    ext_rank['external_rank'] = np.arange(1, len(ext_rank) + 1)

    disc = discovery_driver_ranking[['driver', 'global_rank', 'mean_score', 'evidence_score']].copy() if 'evidence_score' in discovery_driver_ranking.columns else discovery_driver_ranking[['driver', 'global_rank', 'mean_score']].copy()
    merged = disc.merge(ext_rank, on='driver', how='inner')
    merged['rank_shift'] = merged['external_rank'] - merged['global_rank']
    top_k = int(min(max(10, int(np.ceil(len(disc) * 0.25))), max(len(disc), 1)))
    merged['discovery_core_driver'] = merged['global_rank'] <= top_k
    merged['external_core_driver'] = merged['external_rank'] <= top_k
    merged['core_support'] = merged['discovery_core_driver'] & (merged['mean_activity_z'] >= 0.25) & merged.get('coverage_pass', True)
    merged['agreement_label'] = np.where(
        merged['core_support'], 'core_supported',
        np.where(merged['discovery_core_driver'], 'core_not_supported', 'noncore')
    )
    if len(merged) >= 2:
        rho_all, p_all = spearmanr(merged['global_rank'], merged['external_rank'])
    else:
        rho_all, p_all = np.nan, np.nan
    top_disc = merged[merged['discovery_core_driver']].copy()
    if len(top_disc) >= 2:
        rho_core, p_core = spearmanr(top_disc['global_rank'], top_disc['external_rank'])
    else:
        rho_core, p_core = np.nan, np.nan

    core = merged[merged['discovery_core_driver']].copy()
    noncore = merged[~merged['discovery_core_driver']].copy()
    summary_rows = [{
        'discovery_core_k': int(top_k),
        'n_drivers_compared': int(len(merged)),
        'n_core_drivers': int(len(core)),
        'n_core_supported': int(core['core_support'].sum()) if not core.empty else 0,
        'core_support_rate': float(core['core_support'].mean()) if not core.empty else np.nan,
        'mean_core_activity_z': float(core['mean_activity_z'].mean()) if not core.empty else np.nan,
        'mean_noncore_activity_z': float(noncore['mean_activity_z'].mean()) if not noncore.empty else np.nan,
        'delta_core_vs_noncore_activity_z': float(core['mean_activity_z'].mean() - noncore['mean_activity_z'].mean()) if (not core.empty and not noncore.empty) else np.nan,
        'spearman_rho_all': float(rho_all) if pd.notna(rho_all) else np.nan,
        'spearman_pvalue_all': float(p_all) if pd.notna(p_all) else np.nan,
        'spearman_rho_core': float(rho_core) if pd.notna(rho_core) else np.nan,
        'spearman_pvalue_core': float(p_core) if pd.notna(p_core) else np.nan,
        'coverage_pass_rate': float(merged['coverage_pass'].mean()) if 'coverage_pass' in merged.columns and len(merged) else np.nan,
    }]
    summary_df = pd.DataFrame(summary_rows)

    merged.attrs['spearman_rho'] = rho_all
    merged.attrs['spearman_pvalue'] = p_all
    merged.attrs['core_spearman_rho'] = rho_core
    merged.attrs['core_spearman_pvalue'] = p_core
    merged.attrs['discovery_core_k'] = top_k
    return {
        'available': True,
        'sample_summaries': pd.DataFrame(sample_rows),
        'driver_replication': merged.sort_values(['discovery_core_driver', 'core_support', 'agreement_score'], ascending=[False, False, False]).reset_index(drop=True),
        'program_activity': program_activity.sort_values(['driver', 'sample']).reset_index(drop=True),
        'driver_expression': driver_expr.sort_values(['driver', 'sample']).reset_index(drop=True),
        'summary': summary_df,
    }


def orthogonal_xenium_validation(driver_programs: pd.DataFrame, xenium_root: str | Path) -> dict:
    xenium_root = Path(xenium_root)
    runs = list_xenium_runs(xenium_root)
    if xenium_root.exists() and _find_first(xenium_root, ['*cell_feature_matrix.h5']) is not None and _find_first(xenium_root, ['*cells.csv.gz', '*cells.csv']) is not None:
        runs = [xenium_root] + [r for r in runs if r != xenium_root]
    panel_path = _find_first(xenium_root, ['*gene_panel.json.gz', '*gene_panel.json'])
    panel_genes = [g.upper() for g in read_xenium_gene_panel_json(panel_path)] if panel_path else []
    panel_set = set(panel_genes)
    coverage_rows = []
    for driver, df in driver_programs.groupby('driver'):
        targets = [str(x).upper() for x in df['target']]
        overlap = sorted(panel_set.intersection(set(targets + [str(driver).upper()])))
        coverage_rows.append({
            'driver': driver,
            'n_program_genes': len(set(targets)),
            'n_panel_overlap': len(overlap),
            'panel_coverage_rate': len(overlap) / max(len(set(targets + [str(driver).upper()])), 1),
            'overlap_genes': ';'.join(overlap),
        })
    coverage = pd.DataFrame(coverage_rows).sort_values(['n_panel_overlap', 'driver'], ascending=[False, True]) if coverage_rows else pd.DataFrame()

    run_rows = []
    cell_program_rows = []
    if runs:
        request_genes = set(driver_programs['target'].astype(str).tolist()) | set(driver_programs['driver'].astype(str).tolist())
        for run in runs:
            run_name = _archive_sample_name(run) if run.is_file() else ('root_extracted_run' if run == xenium_root else run.name)
            try:
                expr, coords = read_xenium_run(run, genes=request_genes, prefix=f'{run_name}|')
            except Exception:
                run_rows.append({'run': run_name, 'status': 'coverage_only'})
                continue
            run_rows.append({'run': run_name, 'status': 'loaded', 'n_cells': int(expr.shape[0]), 'n_genes': int(expr.shape[1])})
            for driver, df in driver_programs.groupby('driver'):
                score = _safe_program_score(expr, df)
                raw_summ = _program_activity_summary(expr, df)
                cell_program_rows.append({
                    'run': run_name,
                    'driver': driver,
                    'mean_program_score': float(score.mean()),
                    'top10pct_score_cutoff': float(score.quantile(0.90)),
                    'program_activity_raw': float(raw_summ['activity']),
                    'program_activity_mean_raw': float(raw_summ['mean_raw']),
                    'program_activity_iqr_raw': float(raw_summ['iqr_raw']),
                    'n_targets_measured': int(sum(t in expr.columns for t in df['target'])),
                })
    program_scores = pd.DataFrame(cell_program_rows)
    run_level_support = summarize_xenium_run_level_support(program_scores)
    loaded_runs = int((pd.DataFrame(run_rows).get('status', pd.Series(dtype=str)) == 'loaded').sum()) if run_rows else 0
    validation_status = 'three_run_high_resolution' if loaded_runs >= 3 else ('partial_run_level' if loaded_runs > 0 else 'panel_or_coverage_only')
    return {
        'available': xenium_root.exists(),
        'panel_path': str(panel_path) if panel_path else None,
        'panel_gene_count': len(panel_genes),
        'coverage': coverage.reset_index(drop=True) if isinstance(coverage, pd.DataFrame) else pd.DataFrame(),
        'run_summaries': pd.DataFrame(run_rows),
        'program_scores': program_scores,
        'driver_support': run_level_support,
        'run_level_status': pd.DataFrame([{'n_loaded_runs': loaded_runs, 'validation_status': validation_status, 'three_run_validation_ready': bool(loaded_runs >= 3)}]),
    }



def summarize_xenium_run_level_support(program_scores: pd.DataFrame) -> pd.DataFrame:
    """Summarize run-level Xenium support for each discovery driver program."""
    cols = ['driver', 'n_runs_tested', 'n_runs_supported', 'mean_run_activity_z', 'run_consistency', 'mean_targets_measured', 'three_run_ready_driver']
    if program_scores is None or program_scores.empty:
        return pd.DataFrame(columns=cols)
    df = program_scores.copy()
    if 'program_activity_raw' not in df.columns:
        return pd.DataFrame(columns=cols)
    df['program_activity_z'] = df.groupby('run')['program_activity_raw'].transform(
        lambda s: ((s - s.mean()) / (s.std(ddof=0) if float(s.std(ddof=0)) > 1e-12 else 1.0)).astype(float)
    )
    if 'n_targets_measured' not in df.columns:
        df['n_targets_measured'] = 0
    df['run_support_flag'] = (df['n_targets_measured'].fillna(0).astype(int) >= 2) & (df['program_activity_z'].fillna(0.0) > 0)
    out = df.groupby('driver').agg(
        n_runs_tested=('run', 'nunique'),
        n_runs_supported=('run_support_flag', 'sum'),
        mean_run_activity_z=('program_activity_z', 'mean'),
        run_consistency=('run_support_flag', 'mean'),
        mean_targets_measured=('n_targets_measured', 'mean'),
    ).reset_index()
    out['three_run_ready_driver'] = (out['n_runs_tested'].fillna(0).astype(int) >= 3) & (out['mean_targets_measured'].fillna(0) >= 2)
    return out.sort_values(['run_consistency', 'mean_run_activity_z', 'mean_targets_measured'], ascending=[False, False, False]).reset_index(drop=True)[cols]


def maybe_extension_audit(extension_root: str | Path) -> dict:
    extension_root = Path(extension_root)
    if not extension_root.exists():
        return {'available': False, 'files': pd.DataFrame()}
    rows = []
    for sample_dir in sorted(extension_root.iterdir()):
        if not sample_dir.is_dir():
            continue
        present = {k: any(sample_dir.rglob(pattern) for pattern in patterns) for k, patterns in {
            'barcodes': ['*barcodes.tsv.gz', '*barcodes.tsv'],
            'features': ['*features.tsv.gz', '*features.tsv'],
            'matrix': ['*matrix.mtx.gz', '*matrix.mtx'],
            'positions': ['*tissue_positions.csv.gz', '*tissue_positions.csv', '*tissue_positions_list.csv.gz', '*tissue_positions_list.csv'],
            'scalefactors': ['*scalefactors*.json.gz', '*scalefactors*.json'],
            'lowres_image': ['*tissue_lowres_image.png.gz', '*tissue_lowres_image.png'],
            'hires_image': ['*tissue_hires_image.png.gz', '*tissue_hires_image.png'],
        }.items()}
        rows.append({'sample': sample_dir.name, **present, 'qc_pass_minimal': all(present[k] for k in ['barcodes', 'features', 'matrix', 'positions'])})
    return {'available': True, 'files': pd.DataFrame(rows)}
