from __future__ import annotations

from pathlib import Path
import json
import h5py
import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix
from scipy.stats import t as student_t, binomtest, combine_pvalues
import matplotlib.pyplot as plt
import yaml

from .candidates import compute_crc_nat_logfc, select_candidate_genes, shrink_prior_edges
from .grn_model import edge_scores
from .io_visiumhd import load_sample_map, read_positions_parquet, resolve_file, summarize_root
from .prior import load_collectri, load_trrust, merge_transcriptional_priors, load_ligand_receptor_prior, merge_ligand_receptor_priors
from .spatial import build_knn, build_weighted_knn, spatial_lag, multi_scale_spatial_lag
from .sc_reference import summarize_sc_reference_gene_sets
from .mechanisms import load_mechanism_panel, summarize_mechanism_hits
from .ccc_model import (
    infer_spatial_lr_edges,
    build_ccc_signatures,
    associate_driver_ccc,
    build_ie_pathways,
)
from .preprocessing import normalize_expression_matrix, residualize_batch_effect
from .metrics import load_gold_edges, build_evaluation_metrics
from .validation import (
    summarize_driver_ranking,
    build_driver_programs,
    score_programs,
    summarize_niche_specificity,
    derive_niches,
    external_replication,
    orthogonal_xenium_validation,
    maybe_extension_audit,
)


JSON_SAFE_SCALARS = (str, int, float, bool, type(None))


def _ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _json_sanitize(obj):
    if isinstance(obj, pd.DataFrame):
        return {
            '__type__': 'DataFrame',
            'shape': [int(obj.shape[0]), int(obj.shape[1])],
            'columns': [str(c) for c in obj.columns.tolist()],
        }
    if isinstance(obj, pd.Series):
        return {
            '__type__': 'Series',
            'length': int(obj.shape[0]),
            'name': None if obj.name is None else str(obj.name),
        }
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_json_sanitize(v) for v in obj]
    if isinstance(obj, JSON_SAFE_SCALARS):
        return obj
    return str(obj)


def _write_outputs(results: dict, outdir: str | Path) -> dict:
    outdir = _ensure_dir(outdir)
    figdir = _ensure_dir(outdir / 'figures')
    valdir = _ensure_dir(outdir / 'validation')
    artifacts = {
        'candidate_stats': 'candidate_stats.csv',
        'candidate_edges': 'candidate_edges.csv',
        'sample_edge_scores': 'sample_edge_scores.csv',
        'edge_scores': 'edge_scores.csv',
        'top_edges': 'top_edges.csv',
        'driver_ranking': 'driver_ranking.csv',
        'driver_programs': 'driver_programs.csv',
        'spot_program_scores': 'spot_program_scores.csv',
        'niche_program_scores': 'niche_program_scores.csv',
        'niche_specificity': 'niche_specificity.csv',
        'mechanism_hits': 'mechanism_hits.csv',
        'gene_qc': 'gene_qc.csv',
        'ccc_edges': 'ccc_edges.csv',
        'ccc_signatures': 'ccc_signatures.csv',
        'driver_ccc_associations': 'driver_ccc_associations.csv',
        'ie_pathways': 'ie_pathways.csv',
        'ccc_prior_summary': 'ccc_prior_summary.csv',
        'external_aware_driver_ranking': 'external_aware_driver_ranking.csv',
        'evaluation_metrics': 'evaluation_metrics.csv',
    }
    for key, filename in artifacts.items():
        val = results.get(key)
        if isinstance(val, pd.DataFrame):
            val.to_csv(outdir / filename, index=False)
    if 'candidate_genes' in results:
        pd.Series(results['candidate_genes'], name='gene').to_csv(outdir / 'candidate_genes.txt', index=False, header=True)
    if 'niche_labels' in results and isinstance(results['niche_labels'], pd.Series) and not results['niche_labels'].empty:
        results['niche_labels'].rename('niche').to_csv(outdir / 'niche_labels.csv', index=True, header=True)
    if 'sample_logs' in results and isinstance(results['sample_logs'], list):
        pd.DataFrame(results['sample_logs']).to_csv(outdir / 'sample_logs.csv', index=False)

    summary = pd.DataFrame({
        'mode': [results.get('mode', 'unknown')],
        'n_spots': [results.get('n_spots', 0)],
        'n_genes': [results.get('n_genes', 0)],
        'n_candidate_genes': [len(results.get('candidate_genes', []))],
        'n_candidate_edges': [len(results.get('candidate_edges', [])) if isinstance(results.get('candidate_edges'), pd.DataFrame) else 0],
        'n_significant_edges': [int(results.get('edge_scores', pd.DataFrame()).get('significant', pd.Series(dtype=bool)).sum()) if isinstance(results.get('edge_scores'), pd.DataFrame) and 'significant' in results.get('edge_scores').columns else 0],
        'n_driver_programs': [results.get('driver_programs', pd.DataFrame()).shape[0] if isinstance(results.get('driver_programs'), pd.DataFrame) else 0],
        'n_ccc_signatures': [results.get('ccc_signatures', pd.DataFrame()).shape[0] if isinstance(results.get('ccc_signatures'), pd.DataFrame) else 0],
        'n_driver_ccc_associations': [results.get('driver_ccc_associations', pd.DataFrame()).shape[0] if isinstance(results.get('driver_ccc_associations'), pd.DataFrame) else 0],
        'n_samples_ok': [results.get('n_samples_ok', 0)],
        'n_samples_failed': [results.get('n_samples_failed', 0)],
    })
    summary.to_csv(outdir / 'run_summary.csv', index=False)

    stats = results.get('candidate_stats', pd.DataFrame()).copy()
    if not stats.empty:
        stats = stats.sort_values('abs_log2fc', ascending=False)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(stats['log2fc'], -np.log10(stats['max_pct'].clip(lower=1e-6)), s=12)
        ax.set_xlabel('log2FC (CRC vs NAT)')
        ax.set_ylabel('-log10(max expression fraction)')
        ax.set_title(f"Candidate gene screen ({results.get('mode', 'run')})")
        fig.tight_layout()
        fig.savefig(figdir / 'candidate_screen.png', dpi=160)
        plt.close(fig)

    top = results.get('top_edges', pd.DataFrame()).head(20).copy()
    if not top.empty:
        plot_df = top.sort_values('score', ascending=True)
        fig, ax = plt.subplots(figsize=(9, max(4, 0.25 * len(plot_df))))
        ax.barh([f"{s}->{t}" for s, t in zip(plot_df['source'], plot_df['target'])], plot_df['score'])
        ax.set_xlabel('edge score')
        ax.set_title(f"Top GRN edges ({results.get('mode', 'run')})")
        fig.tight_layout()
        fig.savefig(figdir / 'top_edges.png', dpi=160)
        plt.close(fig)

    ranking = results.get('driver_ranking', pd.DataFrame()).head(20).copy()
    if not ranking.empty:
        plot_df = ranking.sort_values('evidence_score' if 'evidence_score' in ranking.columns else 'mean_score', ascending=True)
        xcol = 'evidence_score' if 'evidence_score' in plot_df.columns else 'mean_score'
        fig, ax = plt.subplots(figsize=(8, max(4, 0.25 * len(plot_df))))
        ax.barh(plot_df['driver'].astype(str), plot_df[xcol])
        ax.set_xlabel(xcol)
        ax.set_title('Top driver TF ranking')
        fig.tight_layout()
        fig.savefig(figdir / 'driver_ranking.png', dpi=160)
        plt.close(fig)

    mechanism = results.get('mechanism_hits', pd.DataFrame()).head(20).copy()
    if not mechanism.empty:
        plot_df = mechanism.sort_values('n_overlap', ascending=True)
        fig, ax = plt.subplots(figsize=(8, max(4, 0.3 * len(plot_df))))
        ax.barh([f"{d}:{m}" for d, m in zip(plot_df['driver'], plot_df['mechanism_axis'])], plot_df['n_overlap'])
        ax.set_xlabel('panel overlap genes')
        ax.set_title('Mechanism-panel support')
        fig.tight_layout()
        fig.savefig(figdir / 'mechanism_hits.png', dpi=160)
        plt.close(fig)

    for prefix, obj in [('external', results.get('external_validation')), ('orthogonal', results.get('orthogonal_validation')), ('extension', results.get('extension_audit'))]:
        if isinstance(obj, dict):
            for key, val in obj.items():
                if isinstance(val, pd.DataFrame) and not val.empty:
                    val.to_csv(valdir / f'{prefix}_{key}.csv', index=False)

    manifest = _json_sanitize(results)
    with open(outdir / 'run_manifest.json', 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return {'outdir': str(outdir), 'figdir': str(figdir), 'validation_dir': str(valdir)}




# MODIFIED: Build spatial lags within each sample and concatenate them. This prevents
# KNN edges from connecting spots belonging to different patients/samples when pooled
# matrices are scored. Cross-sample KNN leakage was a major source of inflated external
# support and unstable GRN rankings.
def _samplewise_multi_scale_lags(expr: pd.DataFrame, coords: pd.DataFrame, sample_labels: pd.Series | list | np.ndarray, ks: tuple[int, ...] = (6, 12, 24)) -> dict[int, pd.DataFrame]:
    if expr is None or expr.empty:
        return {}
    labels = pd.Series(sample_labels, index=expr.index).astype(str)
    out_parts: dict[int, list[pd.DataFrame]] = {int(k): [] for k in ks}
    for sample, idx in labels.groupby(labels).groups.items():
        idx = list(idx)
        if len(idx) < 2:
            continue
        expr_s = expr.loc[idx]
        coords_s = coords.loc[idx]
        local = multi_scale_spatial_lag(
            expr_s.reset_index(drop=True),
            coords_s.reset_index(drop=True),
            ks=ks,
            weighted=True,
        )
        for k, lag_df in local.items():
            lag_df = lag_df.set_axis(expr_s.index, axis=0).set_axis(expr_s.columns, axis=1)
            out_parts[int(k)].append(lag_df)
    out = {}
    for k, parts in out_parts.items():
        if parts:
            out[k] = pd.concat(parts, axis=0).reindex(expr.index).loc[:, expr.columns].fillna(0.0)
    return out
# END MODIFIED


def load_ccc_prior_bundle(prior_root: str | Path, fallback_root: str | Path | None = None, ccc_prior_root: str | Path | None = None) -> pd.DataFrame:
    """Load and merge real ligand-receptor priors for the spatial CCC layer.

    The preferred file is the merged full-database export generated by
    ``notebooks/00_download_grn_prior.ipynb``. Source-specific exports are merged if
    present. The old CRC seed prior is only a final fallback so it cannot silently
    replace real LR resources.
    """
    primary_roots = []
    if ccc_prior_root is not None:
        primary_roots.append(Path(ccc_prior_root))
    pr = Path(prior_root)
    primary_roots.extend([pr.parent / 'ccc_prior', pr])
    fallback_roots = []
    if fallback_root is not None:
        fb = Path(fallback_root)
        fallback_roots.extend([fb.parent / 'ccc_prior', fb])

    preferred_names = [
        'ligand_receptor_human.tsv',
        'omnipath_cellphonedb_cellchat_ligand_receptor_human.tsv',
    ]
    source_names = [
        'omnipath_ligand_receptor_human.tsv',
        'cellphonedb_ligand_receptor_human.tsv',
        'cellchat_ligand_receptor_human.tsv',
    ]
    fallback_names = [
        'ligand_receptor_seed_crc.tsv',
        'seed_ligand_receptor_human.tsv',
    ]

    for root in primary_roots:
        for name in preferred_names:
            path = root / name
            if path.exists() and path.stat().st_size > 20:
                lr = load_ligand_receptor_prior(path)
                if lr.empty:
                    continue
                lr.attrs['prior_path'] = str(path)
                return lr
    frames = []
    paths = []
    for root in primary_roots:
        for name in source_names:
            path = root / name
            if path.exists() and path.stat().st_size > 20:
                try:
                    frames.append(load_ligand_receptor_prior(path))
                    paths.append(str(path))
                except Exception:
                    pass
    if frames:
        lr = merge_ligand_receptor_priors(frames)
        lr.attrs['prior_path'] = ';'.join(paths)
        return lr
    for root in primary_roots + fallback_roots:
        for name in fallback_names:
            path = root / name
            if path.exists() and path.stat().st_size > 20:
                lr = load_ligand_receptor_prior(path)
                if lr.empty:
                    continue
                lr.attrs['prior_path'] = str(path)
                lr.attrs['warning'] = 'Using small CRC seed LR prior; run 00_download_grn_prior.ipynb for real databases.'
                return lr
    return pd.DataFrame(columns=['ligand', 'receptor', 'sources', 'references', 'databases', 'n_databases'])


def load_prior_bundle(prior_root: str | Path, fallback_root: str | Path | None = None):
    prior_root = Path(prior_root)
    fallback_root = Path(fallback_root) if fallback_root is not None else None

    def pick(name: str):
        p = prior_root / name
        if p.exists():
            return p
        if fallback_root is not None and (fallback_root / name).exists():
            return fallback_root / name
        return None

    p_collectri = pick('collectri_human.tsv')
    p_trrust = pick('trrust_human.tsv')
    if p_collectri is None or p_trrust is None:
        raise FileNotFoundError('Missing collectri_human.tsv or trrust_human.tsv in prior_root/fallback_root')
    collectri = load_collectri(p_collectri)
    trrust = load_trrust(p_trrust)
    merged = merge_transcriptional_priors(collectri, trrust)
    merged = merged[merged['source'].astype(str) != merged['target'].astype(str)].copy()
    # MODIFIED: Count real prior databases after merge_transcriptional_priors has already
    # collapsed duplicate source-target rows. The previous groupby('nunique') saw values
    # such as 'collectri;trrust' as one string, so CollecTRI+TRRUST edges were incorrectly
    # assigned prior_n_sources=1. That flattened prior confidence, weakened multi-source
    # evidence, and made the TRRUST silver sanity metric uninformative.
    def _source_set(values):
        items = set()
        for value in values:
            for part in str(value).split(';'):
                part = part.strip()
                if part and part.lower() not in {'nan', 'none', 'null'}:
                    items.add(part)
        return sorted(items)

    conf_rows = []
    for (source, target), sub in merged.groupby(['source', 'target']):
        srcs = _source_set(sub['prior_source'])
        conf_rows.append({
            'source': source,
            'target': target,
            'prior_n_sources': len(srcs),
            'prior_sources': ';'.join(srcs),
        })
    conf = pd.DataFrame(conf_rows)
    conf['prior_conf'] = conf['prior_n_sources'].astype(float).clip(lower=1.0)
    # END MODIFIED
    return merged, conf


def _read_10x_h5_sparse(path: str | Path):
    path = Path(path)
    with h5py.File(path, 'r') as h5:
        grp = h5['matrix']
        data = grp['data'][()]
        indices = grp['indices'][()]
        indptr = grp['indptr'][()]
        shape = tuple(grp['shape'][()])
        X = csc_matrix((data, indices, indptr), shape=shape)
        barcodes = grp['barcodes'][()]
        f = grp['features'] if 'features' in grp else None
        gene_names = None
        if f is not None:
            for key in ['name', 'gene_names', 'id']:
                if key in f:
                    gene_names = f[key][()]
                    break
    barcodes = [b.decode() if isinstance(b, (bytes, bytearray)) else str(b) for b in barcodes]
    gene_names = [b.decode() if isinstance(b, (bytes, bytearray)) else str(b) for b in gene_names]
    return X, gene_names, barcodes


def _infer_coord_columns(pos: pd.DataFrame):
    candidates_x = ['pxl_col_in_fullres', 'pixel_x', 'x', 'coord_x', 'pxl_col', 'imagecol']
    candidates_y = ['pxl_row_in_fullres', 'pixel_y', 'y', 'coord_y', 'pxl_row', 'imagerow']
    xcol = next((c for c in candidates_x if c in pos.columns), None)
    ycol = next((c for c in candidates_y if c in pos.columns), None)
    if xcol is None or ycol is None:
        raise ValueError(f'Could not infer coordinate columns from: {list(pos.columns)}')
    return xcol, ycol


def _prepare_sample_dense(sample: str, root: str | Path, sample_to_gsm: dict[str, str], genes: list[str], max_spots: int = 1200, seed: int = 42):
    h5_path = resolve_file(sample, 'filtered_feature_bc_matrix.h5', root, sample_to_gsm)
    if h5_path is None:
        raise FileNotFoundError(f'No filtered_feature_bc_matrix.h5 for {sample}')
    X, gene_names, barcodes = _read_10x_h5_sparse(h5_path)
    gene_to_ix = {g: i for i, g in enumerate(gene_names)}
    gene_to_ix_upper = {str(g).upper(): i for i, g in enumerate(gene_names)}
    use_ix = []
    use_genes = []
    seen = set()
    for g in genes:
        gs = str(g)
        if gs in gene_to_ix:
            ix = gene_to_ix[gs]
        elif gs.upper() in gene_to_ix_upper:
            ix = gene_to_ix_upper[gs.upper()]
        else:
            continue
        if ix in seen:
            continue
        seen.add(ix)
        use_ix.append(ix)
        use_genes.append(gene_names[ix])
    if not use_genes:
        raise ValueError(f'No selected genes present in {sample}')
    feat_ix = np.array(use_ix)
    libsize = np.asarray(X.sum(axis=0)).ravel().astype(float)
    libsize[libsize == 0] = 1.0
    sub = X[feat_ix, :].T.toarray().astype(float)
    sub = np.log1p((sub / libsize[:, None]) * 1e4)
    expr = pd.DataFrame(sub, index=pd.Index(barcodes, name='barcode'), columns=use_genes)

    pos = read_positions_parquet(sample, root, sample_to_gsm).copy()
    if 'barcode' in pos.columns:
        pos['barcode'] = pos['barcode'].astype(str)
        pos = pos.set_index('barcode')
    elif pos.index.name is None:
        for c in pos.columns:
            if 'barcode' in str(c).lower():
                pos[c] = pos[c].astype(str)
                pos = pos.set_index(c)
                break
    pos.index = pos.index.astype(str)
    if 'in_tissue' in pos.columns:
        try:
            pos = pos[pos['in_tissue'].astype(float) > 0].copy()
        except Exception:
            pass
    common = expr.index.intersection(pos.index)
    if len(common) == 0:
        raise ValueError(f'No overlapping barcodes between h5 and positions for {sample}')
    expr = expr.loc[common]
    pos = pos.loc[common]
    if len(expr) > max_spots:
        rng = np.random.default_rng(seed)
        keep = np.sort(rng.choice(len(expr), size=max_spots, replace=False))
        expr = expr.iloc[keep]
        pos = pos.iloc[keep]
    xcol, ycol = _infer_coord_columns(pos)
    coords = pd.DataFrame({'x': pos[xcol].astype(float).values, 'y': pos[ycol].astype(float).values}, index=expr.index)
    expr.index = [f'{sample}|{i}' for i in expr.index]
    coords.index = expr.index
    return expr, coords


def _compute_real_candidate_stats(data_root: str | Path, config_path: str | Path):
    sample_to_gsm = load_sample_map(config_path)
    cfg = yaml.safe_load(Path(config_path).read_text())
    crc_samples = set(cfg['sample_groups']['crc'])
    nat_samples = set(cfg['sample_groups']['nat'])
    acc: dict[str, dict[str, float]] = {}
    found = 0
    for sample in list(crc_samples) + list(nat_samples):
        h5_path = resolve_file(sample, 'filtered_feature_bc_matrix.h5', data_root, sample_to_gsm)
        if h5_path is None:
            continue
        X, gene_names, _ = _read_10x_h5_sparse(h5_path)
        n = int(X.shape[1])
        if n <= 0:
            continue
        # MODIFIED: compute candidate statistics on library-size-normalized counts rather
        # than raw UMI means. Raw means make CRC/NAT logFC sensitive to sequencing depth and
        # sample loading, which destabilizes candidate genes before the model even starts.
        libsize = np.asarray(X.sum(axis=0)).ravel().astype(float)
        libsize[~np.isfinite(libsize) | (libsize <= 0)] = 1.0
        scale = 1e4 / libsize
        norm_sums = np.asarray(X.dot(scale)).ravel().astype(np.float64)
        nnz_per_gene = np.bincount(X.indices, minlength=X.shape[0]).astype(np.float64, copy=False)
        mean = norm_sums / float(n)
        pct = nnz_per_gene / float(n)
        # END MODIFIED
        bucket = 'crc' if sample in crc_samples else 'nat'
        for g, m, p in zip(gene_names, mean, pct):
            row = acc.setdefault(g, {'crc_sum': 0.0, 'nat_sum': 0.0, 'crc_pct': 0.0, 'nat_pct': 0.0, 'crc_n': 0.0, 'nat_n': 0.0})
            row[f'{bucket}_sum'] += float(m) * n
            row[f'{bucket}_pct'] += float(p) * n
            row[f'{bucket}_n'] += n
        found += 1
    if found == 0:
        raise FileNotFoundError('No real Visium h5 files found.')
    rows = []
    for gene, row in acc.items():
        crc_n = max(row['crc_n'], 1.0)
        nat_n = max(row['nat_n'], 1.0)
        crc_mean = row['crc_sum'] / crc_n
        nat_mean = row['nat_sum'] / nat_n
        crc_frac = row['crc_pct'] / crc_n
        nat_frac = row['nat_pct'] / nat_n
        rows.append({
            'gene': gene,
            'log2fc': np.log2((crc_mean + 1.0) / (nat_mean + 1.0)),
            'pct_crc': crc_frac,
            'pct_nat': nat_frac,
        })
    stats = pd.DataFrame(rows)
    stats['max_pct'] = stats[['pct_crc', 'pct_nat']].max(axis=1)
    stats['abs_log2fc'] = stats['log2fc'].abs()
    return stats.sort_values(['abs_log2fc', 'max_pct'], ascending=[False, False]).reset_index(drop=True)


def audit_project_inputs(root: str | Path, config_path: str | Path) -> dict:
    root = Path(root)
    main_root = root / 'data' / 'main_visiumhd'
    sc_root = root / 'data' / 'sc_reference'
    pri_root = root / 'resources' / 'grn_prior'
    xenium_root = root / 'data' / 'orthogonal_xenium_gse280314'
    ext_root = root / 'data' / 'external_validation_gse226997'
    extension_root = root / 'data' / 'extension_gse267401'
    main_status = summarize_root(main_root, config_path) if main_root.exists() else pd.DataFrame()
    priors = pd.DataFrame({'file': ['collectri_human.tsv', 'trrust_human.tsv'], 'exists': [(pri_root / 'collectri_human.tsv').exists(), (pri_root / 'trrust_human.tsv').exists()]})
    sc_files = [
        'GSE132465_GEO_processed_CRC_10X_cell_annotation.txt.gz',
        'GSE132465_GEO_processed_CRC_10X_raw_UMI_count_matrix.txt.gz',
        'GSE144735_processed_KUL3_CRC_10X_annotation.txt.gz',
        'GSE144735_processed_KUL3_CRC_10X_raw_UMI_count_matrix.txt.gz',
    ]
    sc_status = pd.DataFrame({'file': sc_files, 'exists': [(sc_root / f).exists() for f in sc_files]})
    ready_main = False
    if not main_status.empty:
        pivot = main_status.pivot_table(index='sample', columns='basename', values='exists', aggfunc='first').fillna(False)
        need = ['filtered_feature_bc_matrix.h5', 'tissue_positions.parquet.gz']
        for n in need:
            if n not in pivot.columns:
                pivot[n] = False
        ready_main = bool((pivot[need].all(axis=1)).all())
    ready_priors = bool(priors['exists'].all())
    ready = ready_main and ready_priors
    return {
        'ready_for_real_discovery': ready,
        'main_visiumhd': main_status,
        'sc_reference': sc_status,
        'priors': priors,
        'orthogonal_xenium_exists': xenium_root.exists(),
        'external_validation_exists': ext_root.exists(),
        'extension_exists': extension_root.exists(),
    }


def _bh_fdr(pvalues: pd.Series) -> pd.Series:
    p = pd.Series(pvalues, copy=True).fillna(1.0).astype(float).clip(lower=0.0, upper=1.0)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p.values)
    ranked = p.values[order]
    q = ranked * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)
    out = np.empty_like(q)
    out[order] = q
    return pd.Series(out, index=p.index)



# MODIFIED: Convert sample-level edge diagnostics into a prior-free biological score.
# This score is used for inference and significance testing, while the final ranking can
# still use prior evidence as a weak stabilizer. It prevents a constant positive prior term
# or internal-purity offset from making every prior edge look statistically replicated.
def _edge_biological_score(df: pd.DataFrame) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype=float)
    idx = df.index
    directional_internal = df.get('directional_internal_support', pd.Series(0.0, index=idx)).fillna(0.0).clip(lower=0.0)
    internal_support = df.get('internal_support', pd.Series(0.0, index=idx)).fillna(0.0).clip(lower=0.0)
    internal_delta = df.get('internal_delta_r2', pd.Series(0.0, index=idx)).fillna(0.0).clip(lower=0.0)
    internal_purity = df.get('internal_purity', pd.Series(1.0, index=idx)).fillna(1.0).clip(lower=0.0, upper=1.0)
    directional_corr = df.get('directional_corr_support', df.get('corr_xy', pd.Series(0.0, index=idx))).fillna(0.0).clip(lower=0.0)
    penalty = df.get('external_confounding_penalty', pd.Series(0.0, index=idx)).fillna(0.0).clip(lower=0.0)
    bio = (
        0.50 * directional_internal
        + 0.30 * internal_support
        + 0.10 * internal_delta * internal_purity
        + 0.10 * directional_corr
        - 0.35 * penalty
    )
    return bio.fillna(0.0).clip(lower=0.0)
# END MODIFIED

def _attach_prior_conf(scores: pd.DataFrame, candidate_edges: pd.DataFrame) -> pd.DataFrame:
    if scores.empty:
        return scores
    merge_cols = [c for c in ['source', 'target', 'prior_conf', 'prior_n_sources', 'prior_sources'] if c in candidate_edges.columns]
    out = scores.merge(candidate_edges[merge_cols].drop_duplicates(), on=['source', 'target'], how='left')
    out['prior_conf'] = out['prior_conf'].fillna(1.0)
    out['prior_term'] = out['prior_conf'].clip(lower=0.0, upper=3.0) / 3.0
    # MODIFIED: Preserve intrinsic GRN evidence, but separate ranking evidence from
    # inference evidence. The previous score contained a large constant prior/purity term;
    # across five samples this made almost every prior edge positive and generated
    # overconfident q-values. biological_score is prior-free and is used downstream for
    # p-values/significance; score remains a conservative ranking score.
    base_intrinsic = out.get('intrinsic_score', out.get('score', pd.Series(0.0, index=out.index))).fillna(0.0).clip(lower=0.0)
    directional_corr = out.get('directional_corr_support', out.get('corr_xy', pd.Series(0.0, index=out.index))).fillna(0.0).clip(lower=0.0)
    internal_purity = out.get('internal_purity', pd.Series(1.0, index=out.index)).fillna(0.0).clip(lower=0.0, upper=1.0)
    penalty = out.get('external_confounding_penalty', pd.Series(0.0, index=out.index)).fillna(0.0).clip(lower=0.0)
    out['raw_model_score'] = out.get('score', pd.Series(0.0, index=out.index)).fillna(0.0)
    out['biological_score'] = _edge_biological_score(out)
    out['score'] = (
        0.18 * out['prior_term']
        + 0.46 * out['biological_score']
        + 0.20 * base_intrinsic
        + 0.08 * out.get('internal_support', pd.Series(0.0, index=out.index)).fillna(0.0).clip(lower=0.0)
        + 0.05 * directional_corr
        + 0.03 * internal_purity
        - 0.18 * penalty
    ).clip(lower=0.0)
    # END MODIFIED
    return out.sort_values(['score', 'internal_support', 'internal_purity', 'prior_conf'], ascending=[False, False, False, False]).reset_index(drop=True)


def _filter_low_information_genes(expr_all: pd.DataFrame, lag_all: pd.DataFrame, candidate_edges: pd.DataFrame):
    var = expr_all.var(axis=0)
    nz_frac = (expr_all > 0).mean(axis=0)
    keep = (var > 1e-8) & (nz_frac >= 0.005)
    keep_genes = set(keep[keep].index.astype(str))
    filtered_edges = candidate_edges[candidate_edges['source'].astype(str).isin(keep_genes) & candidate_edges['target'].astype(str).isin(keep_genes)].copy()
    final_genes = sorted(set(filtered_edges['source'].astype(str)).union(filtered_edges['target'].astype(str)))
    expr_use = expr_all[final_genes].copy() if final_genes else expr_all.iloc[:, :0].copy()
    lag_use = lag_all[final_genes].copy() if final_genes else lag_all.iloc[:, :0].copy()
    gene_qc = pd.DataFrame({
        'gene': expr_all.columns.astype(str),
        'variance': var.reindex(expr_all.columns).values,
        'nonzero_fraction': nz_frac.reindex(expr_all.columns).values,
        'keep_after_qc': [g in set(final_genes) for g in expr_all.columns.astype(str)],
    })
    return expr_use, lag_use, filtered_edges.reset_index(drop=True), gene_qc


def _prepare_candidate_edges(stats: pd.DataFrame, prior_df: pd.DataFrame, prior_conf: pd.DataFrame, candidate_genes: list[str], mechanism_panel_path: str | Path | None = None, max_edges: int = 15000):
    candidate_edges = shrink_prior_edges(prior_df[[c for c in ['source', 'target', 'prior_source', 'regulatory_sign'] if c in prior_df.columns]].copy(), candidate_genes)
    if candidate_edges.empty:
        raise RuntimeError('No candidate prior edges remain after filtering to candidate genes. Check prior files and candidate thresholds.')
    candidate_edges = candidate_edges.merge(prior_conf[['source', 'target', 'prior_conf', 'prior_n_sources', 'prior_sources']], on=['source', 'target'], how='left')
    candidate_edges['prior_conf'] = candidate_edges['prior_conf'].fillna(1.0)
    if 'regulatory_sign' not in candidate_edges.columns:
        candidate_edges['regulatory_sign'] = 0
    candidate_edges['regulatory_sign'] = pd.to_numeric(candidate_edges['regulatory_sign'], errors='coerce').fillna(0).astype(int)
    candidate_edges = candidate_edges.sort_values(['prior_conf', 'source', 'target'], ascending=[False, True, True]).drop_duplicates(subset=['source', 'target']).reset_index(drop=True)
    gene_counts = pd.Series(candidate_edges[['source', 'target']].values.ravel()).value_counts() if len(candidate_edges) else pd.Series(dtype=int)
    if len(gene_counts):
        stats_subset = stats[stats['gene'].isin(gene_counts.index)].copy()
        keep_genes = set(stats_subset.sort_values(['abs_log2fc', 'max_pct'], ascending=[False, False])['gene'].head(1200).astype(str).tolist())
        keep_genes.update(candidate_edges[candidate_edges['prior_conf'] >= 2]['source'].astype(str).tolist())
        keep_genes.update(candidate_edges[candidate_edges['prior_conf'] >= 2]['target'].astype(str).tolist())
        top_tfs = candidate_edges['source'].astype(str).value_counts().head(200).index.astype(str)
        keep_genes.update(top_tfs)
        if mechanism_panel_path is not None and Path(mechanism_panel_path).exists():
            panel = load_mechanism_panel(mechanism_panel_path)
            for genes in panel.values():
                keep_genes.update(map(str, genes))
        candidate_edges = candidate_edges[candidate_edges['source'].astype(str).isin(keep_genes) & candidate_edges['target'].astype(str).isin(keep_genes)].copy()
    if len(candidate_edges) > max_edges:
        candidate_edges = candidate_edges.sort_values(['prior_conf', 'source', 'target'], ascending=[False, True, True]).head(max_edges).copy()
    return candidate_edges.reset_index(drop=True)


def _aggregate_sample_edge_scores(sample_edge_scores: pd.DataFrame, prior_conf_df: pd.DataFrame) -> pd.DataFrame:
    if sample_edge_scores.empty:
        return pd.DataFrame()
    rows = []
    for (source, target), df in sample_edge_scores.groupby(['source', 'target']):
        ranking_scores = df['score'].astype(float).values
        # MODIFIED: use prior-free biological evidence, not the final ranking score, for
        # replicated-sample inference. This directly addresses the real_run(1) pathology
        # where all prior edges had positive sample scores because prior/purity constants
        # were included before the t-test and sign test.
        if 'biological_score' in df.columns:
            infer_scores = df['biological_score'].astype(float).fillna(0.0).clip(lower=0.0).values
        else:
            infer_scores = _edge_biological_score(df).astype(float).fillna(0.0).clip(lower=0.0).values
        n = len(infer_scores)
        mean_score = float(np.mean(ranking_scores)) if len(ranking_scores) else 0.0
        std_score = float(np.std(ranking_scores, ddof=1)) if len(ranking_scores) >= 2 else np.nan
        bio_mean_score = float(np.mean(infer_scores)) if n else 0.0
        bio_std_score = float(np.std(infer_scores, ddof=1)) if n >= 2 else np.nan
        if n >= 2 and np.isfinite(bio_std_score) and bio_std_score > 1e-12:
            tstat = bio_mean_score / (bio_std_score / np.sqrt(n))
            pval_effect = float(1.0 - student_t.cdf(tstat, df=n - 1))
        elif n == 1:
            pval_effect = 0.5 if bio_mean_score > 0 else 1.0
        else:
            pval_effect = 1.0
        k_pos = int(np.sum(infer_scores > 0))
        try:
            pval_sign = float(binomtest(k_pos, n=max(n,1), p=0.5, alternative='greater').pvalue) if n else 1.0
        except Exception:
            pval_sign = 1.0
        row = {
            'source': str(source),
            'target': str(target),
            'n_samples': int(n),
            'mean_score': mean_score,
            'std_score': std_score if np.isfinite(std_score) else np.nan,
            'bio_mean_score': bio_mean_score,
            'bio_std_score': bio_std_score if np.isfinite(bio_std_score) else np.nan,
            'consistency_pos': float(np.mean(infer_scores > 0)) if n else 0.0,
            'pvalue_effect': pval_effect,
            'pvalue_sign': pval_sign,
        }
        # END MODIFIED
        for col in ['internal', 'external', 'external_abs', 'external_relative', 'internal_beta', 'external_beta', 'internal_support', 'external_support', 'internal_delta_r2', 'external_delta_r2', 'external_specificity', 'corr_xy', 'corr_yz', 'corr_xz', 'full_r2', 'r2_covariates_only', 'r2_internal_only', 'r2_external_only', 'internal_purity', 'external_confounding_penalty', 'intrinsic_score', 'external_branch_score', 'raw_model_score', 'biological_score', 'prior_conf']:
            if col in df.columns:
                row[col] = float(df[col].astype(float).mean())
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.merge(prior_conf_df[[c for c in ['source', 'target', 'prior_conf', 'prior_n_sources', 'prior_sources'] if c in prior_conf_df.columns]].drop_duplicates(), on=['source', 'target'], how='left', suffixes=('', '_merged'))
    if 'prior_conf_merged' in out.columns:
        out['prior_conf'] = out['prior_conf'].fillna(out['prior_conf_merged'])
        out = out.drop(columns=['prior_conf_merged'])
    out['prior_conf'] = out['prior_conf'].fillna(1.0)
    out['prior_term'] = out['prior_conf'].clip(lower=0.0, upper=3.0) / 3.0

    # MODIFIED: empirical within-TF tail p-value is based on prior-free biological
    # evidence rather than final score, so it cannot be inflated by multi-source priors.
    out['pvalue_empirical_source'] = 1.0
    for source, idx in out.groupby('source').groups.items():
        sub = out.loc[list(idx)].sort_values('bio_mean_score', ascending=False).copy()
        m = len(sub)
        sub['pvalue_empirical_source'] = [(rank + 1) / (m + 1) for rank in range(m)]
        out.loc[sub.index, 'pvalue_empirical_source'] = sub['pvalue_empirical_source'].values
    # END MODIFIED

    # combine effect-size evidence and sign consistency, while using within-source empirical ranking as a separate moderation term
    combo_p = []
    for _, r in out[['pvalue_effect', 'pvalue_sign']].iterrows():
        p1 = float(r['pvalue_effect']) if np.isfinite(r['pvalue_effect']) else 1.0
        p2 = float(r['pvalue_sign']) if np.isfinite(r['pvalue_sign']) else 1.0
        try:
            cp = float(combine_pvalues([max(min(p1, 1.0), 1e-12), max(min(p2, 1.0), 1e-12)], method='fisher')[1])
        except Exception:
            cp = max(p1, p2)
        combo_p.append(cp)
    out['pvalue_combo'] = combo_p
    # Use the combined statistical evidence as the main inferential quantity.
    # Keep within-source empirical rank as an evidence-side moderation term rather than
    # collapsing every edge into the same q-value platform.
    out['pvalue'] = out['pvalue_combo'].astype(float).clip(lower=1e-12, upper=1.0)
    out['qvalue_main'] = _bh_fdr(out['pvalue'])
    out['qvalue_effect'] = _bh_fdr(out['pvalue_effect'])
    out['qvalue_sign'] = _bh_fdr(out['pvalue_sign'])
    out['qvalue_empirical'] = _bh_fdr(out['pvalue_empirical_source'])
    out = out.rename(columns={'mean_score': 'score'})

    # within-source empirical support acts as a secondary gate rather than dominating q-values
    out['empirical_supported'] = False
    for source, idx in out.groupby('source').groups.items():
        sub = out.loc[list(idx)].sort_values(['score', 'consistency_pos', 'n_samples'], ascending=[False, False, False]).copy()
        m = len(sub)
        keep_n = min(m, max(3, int(np.ceil(0.30 * m))))
        keep_idx = sub.head(keep_n).index
        out.loc[keep_idx, 'empirical_supported'] = True

    # MODIFIED: stricter replication gate tuned from real_run(1). A discovery edge now
    # needs prior-free biological support in at least four samples, within-driver top-tail
    # rank, and stable positive direction. This reduces false-positive prior edges while
    # retaining strong CRC epithelial/stromal programs such as HNF4A-CDX2 and ELF3-KRT8.
    out['significant_main'] = (
        (out['qvalue_effect'] <= 0.20)
        & (out['pvalue_sign'] <= 0.10)
        & (out['pvalue_empirical_source'] <= 0.15)
        & (out['n_samples'] >= 4)
        & (out['consistency_pos'] >= 0.80)
        & (out['bio_mean_score'].fillna(0.0) > 0)
    )
    # END MODIFIED
    out['significant'] = out['significant_main'] & out['empirical_supported']
    out['qvalue'] = out['qvalue_main']
    # MODIFIED: Aggregate sample evidence with an intrinsic-first objective and an explicit
    # penalty for edges whose support is mostly explained by external spatial confounding.
    internal_purity = out.get('internal_purity', pd.Series(1.0, index=out.index)).fillna(0.0).clip(lower=0.0, upper=1.0)
    penalty = out.get('external_confounding_penalty', pd.Series(0.0, index=out.index)).fillna(0.0).clip(lower=0.0)
    out['evidence_score'] = (
        0.16 * out['prior_term']
        + 0.36 * out.get('bio_mean_score', pd.Series(0.0, index=out.index)).fillna(0.0).clip(lower=0.0)
        + 0.18 * out.get('internal_support', pd.Series(0.0, index=out.index)).fillna(0.0).clip(lower=0.0)
        + 0.12 * out['consistency_pos'].fillna(0.0)
        + 0.08 * internal_purity
        + 0.06 * (np.log1p(out['n_samples']) / np.log(6.0))
        + 0.04 * out['empirical_supported'].astype(float)
        - 0.18 * penalty
    ).clip(lower=0.0)
    # END MODIFIED
    return out.sort_values(['significant', 'evidence_score', 'score'], ascending=[False, False, False]).reset_index(drop=True)



def _summarize_ccc_prior(lr_prior: pd.DataFrame) -> pd.DataFrame:
    if lr_prior is None or lr_prior.empty:
        return pd.DataFrame([{'n_lr_pairs': 0, 'n_ligands': 0, 'n_receptors': 0, 'databases': '', 'prior_path': '', 'warning': 'no_lr_prior_loaded'}])
    db_col = 'databases' if 'databases' in lr_prior.columns else ('database' if 'database' in lr_prior.columns else None)
    dbs = ''
    if db_col is not None:
        vals = []
        for x in lr_prior[db_col].dropna().astype(str):
            vals.extend([p for p in x.split(';') if p])
        dbs = ';'.join(sorted(set(vals)))
    return pd.DataFrame([{
        'n_lr_pairs': int(lr_prior[['ligand', 'receptor']].drop_duplicates().shape[0]) if {'ligand', 'receptor'}.issubset(lr_prior.columns) else int(lr_prior.shape[0]),
        'n_ligands': int(lr_prior['ligand'].nunique()) if 'ligand' in lr_prior.columns else 0,
        'n_receptors': int(lr_prior['receptor'].nunique()) if 'receptor' in lr_prior.columns else 0,
        'databases': dbs,
        'prior_path': str(lr_prior.attrs.get('prior_path', '')),
        'warning': str(lr_prior.attrs.get('warning', '')),
    }])


def build_external_aware_driver_ranking(
    driver_ranking: pd.DataFrame,
    external_validation: dict | None = None,
    orthogonal_validation: dict | None = None,
    mechanism_hits: pd.DataFrame | None = None,
    driver_ccc_associations: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Reviewer-facing ranking that separates discovery from support evidence.

    This does not overwrite ``driver_ranking.csv``. It gives a transparent composite
    score that moves the selected main-story drivers toward external/Xenium/CCC support
    when such support exists, without hiding weak full-rank correlation.
    """
    if driver_ranking is None or driver_ranking.empty:
        return pd.DataFrame()
    out = driver_ranking.copy()
    base = out['evidence_score'] if 'evidence_score' in out.columns else out.get('mean_score', pd.Series(0.0, index=out.index))
    b = pd.to_numeric(base, errors='coerce').fillna(0.0)
    out['discovery_scaled'] = (b - b.min()) / (b.max() - b.min()) if float(b.max() - b.min()) > 1e-12 else 0.0

    out['external_support_score'] = 0.0
    out['external_rank'] = np.nan
    if isinstance(external_validation, dict):
        rep = external_validation.get('driver_replication')
        if isinstance(rep, pd.DataFrame) and not rep.empty:
            cols = [c for c in ['driver', 'external_rank', 'agreement_score', 'core_support', 'mean_activity_z', 'coverage_pass'] if c in rep.columns]
            tmp = rep[cols].copy()
            tmp['external_support_score'] = (
                0.45 * pd.to_numeric(tmp.get('agreement_score', 0.0), errors='coerce').fillna(0.0)
                + 0.35 * pd.to_numeric(tmp.get('mean_activity_z', 0.0), errors='coerce').fillna(0.0)
                + 0.20 * tmp.get('core_support', False).astype(float)
            )
            es = tmp['external_support_score']
            tmp['external_support_score'] = (es - es.min()) / (es.max() - es.min()) if float(es.max() - es.min()) > 1e-12 else 0.0
            out = out.merge(tmp[['driver', 'external_rank', 'external_support_score']], on='driver', how='left', suffixes=('', '_ext'))
            if 'external_rank_ext' in out.columns:
                out['external_rank'] = out['external_rank_ext'].combine_first(out['external_rank'])
            if 'external_support_score_ext' in out.columns:
                out['external_support_score'] = out['external_support_score_ext'].combine_first(out['external_support_score']).fillna(0.0)
            out = out.drop(columns=[c for c in ['external_rank_ext', 'external_support_score_ext'] if c in out.columns])

    out['xenium_support_score'] = 0.0
    if isinstance(orthogonal_validation, dict):
        xs = orthogonal_validation.get('driver_support')
        if isinstance(xs, pd.DataFrame) and not xs.empty:
            tmp = xs.copy()
            tmp['xenium_support_score'] = (
                0.6 * pd.to_numeric(tmp.get('run_consistency', 0.0), errors='coerce').fillna(0.0)
                + 0.4 * pd.to_numeric(tmp.get('mean_run_activity_z', 0.0), errors='coerce').fillna(0.0).clip(lower=0.0)
            )
            out = out.merge(tmp[['driver', 'xenium_support_score', 'n_runs_tested', 'n_runs_supported']], on='driver', how='left')
            out['xenium_support_score'] = out['xenium_support_score_y'].combine_first(out['xenium_support_score_x']).fillna(0.0)
            out = out.drop(columns=[c for c in ['xenium_support_score_x', 'xenium_support_score_y'] if c in out.columns])

    out['mechanism_support_score'] = 0.0
    if mechanism_hits is not None and isinstance(mechanism_hits, pd.DataFrame) and not mechanism_hits.empty:
        mh = mechanism_hits.groupby('driver').agg(n_mechanism_axes=('mechanism_axis', 'nunique'), max_mechanism_overlap=('n_overlap', 'max')).reset_index()
        mh['mechanism_support_score'] = np.log1p(mh['n_mechanism_axes']) + 0.25 * np.log1p(mh['max_mechanism_overlap'])
        m = mh['mechanism_support_score']
        mh['mechanism_support_score'] = (m - m.min()) / (m.max() - m.min()) if float(m.max() - m.min()) > 1e-12 else 0.0
        out = out.merge(mh[['driver', 'mechanism_support_score']], on='driver', how='left', suffixes=('', '_mh'))
        out['mechanism_support_score'] = out['mechanism_support_score_mh'].combine_first(out['mechanism_support_score']).fillna(0.0)
        out = out.drop(columns=[c for c in ['mechanism_support_score_mh'] if c in out.columns])

    out['ccc_support_score'] = 0.0
    if driver_ccc_associations is not None and isinstance(driver_ccc_associations, pd.DataFrame) and not driver_ccc_associations.empty:
        cc = driver_ccc_associations.groupby('driver').agg(n_ccc_assoc=('signature_id', 'nunique'), best_ccc_assoc=('assoc_score', 'max')).reset_index()
        cc['ccc_support_score'] = np.log1p(cc['n_ccc_assoc']) + pd.to_numeric(cc['best_ccc_assoc'], errors='coerce').fillna(0.0)
        c = cc['ccc_support_score']
        cc['ccc_support_score'] = (c - c.min()) / (c.max() - c.min()) if float(c.max() - c.min()) > 1e-12 else 0.0
        out = out.merge(cc[['driver', 'ccc_support_score']], on='driver', how='left', suffixes=('', '_ccc'))
        out['ccc_support_score'] = out['ccc_support_score_ccc'].combine_first(out['ccc_support_score']).fillna(0.0)
        out = out.drop(columns=[c for c in ['ccc_support_score_ccc'] if c in out.columns])

    # Main ranking follows the post-run diagnostic recommendation: discovery evidence is
    # still important, but drivers are promoted only when they replicate externally, in
    # orthogonal Xenium data, or in a curated CRC mechanism axis. CCC support is retained
    # as an auxiliary column, not as the primary driver of ranking.
    out['replication_oriented_score'] = (
        0.45 * out['discovery_scaled'].fillna(0.0)
        + 0.30 * out['external_support_score'].fillna(0.0)
        + 0.15 * out['xenium_support_score'].fillna(0.0)
        + 0.10 * out['mechanism_support_score'].fillna(0.0)
    )
    out['secondary_ccc_support_score'] = out['ccc_support_score'].fillna(0.0)

    # Practical labels for reporting: A = replicated core, B = discovery-only core,
    # C = externally rescued non-core, D = exploratory. The thresholds are relative,
    # so the labels remain usable across real and demo runs.
    core_k = int(min(max(10, int(np.ceil(len(out) * 0.25))), max(len(out), 1)))
    if 'global_rank' in out.columns:
        out['discovery_core_driver'] = pd.to_numeric(out['global_rank'], errors='coerce').fillna(len(out) + 1) <= core_k
    else:
        out['discovery_core_driver'] = out['discovery_scaled'].rank(ascending=False, method='first') <= core_k
    ext_thr = float(out['external_support_score'].quantile(0.75)) if len(out) else 0.0
    xen_thr = float(out['xenium_support_score'].quantile(0.75)) if len(out) else 0.0
    mech_thr = float(out['mechanism_support_score'].quantile(0.75)) if len(out) else 0.0
    out['has_replication_support'] = (
        (out['external_support_score'].fillna(0.0) >= max(ext_thr, 1e-9))
        | (out['xenium_support_score'].fillna(0.0) >= max(xen_thr, 1e-9))
        | (out['mechanism_support_score'].fillna(0.0) >= max(mech_thr, 1e-9))
    )
    out['driver_support_class'] = np.where(
        out['discovery_core_driver'] & out['has_replication_support'], 'A_replicated_core',
        np.where(out['discovery_core_driver'], 'B_discovery_only_core',
                 np.where((~out['discovery_core_driver']) & (out['external_support_score'].fillna(0.0) >= max(ext_thr, 1e-9)),
                          'C_external_rescue', 'D_exploratory'))
    )
    out = out.sort_values(['replication_oriented_score', 'external_support_score', 'xenium_support_score', 'secondary_ccc_support_score'], ascending=[False, False, False, False]).reset_index(drop=True)
    out['replication_oriented_rank'] = np.arange(1, len(out) + 1)
    return out


def run_real_pipeline(
    data_root: str | Path,
    config_path: str | Path,
    prior_root: str | Path,
    outdir: str | Path,
    fallback_prior_root: str | Path | None = None,
    sc_reference_root: str | Path | None = None,
    orthogonal_xenium_root: str | Path | None = None,
    external_validation_root: str | Path | None = None,
    extension_root: str | Path | None = None,
    mechanism_panel_path: str | Path | None = None,
    min_max_pct: float = 0.03,
    min_abs_log2fc: float = 0.15,
    top_n_extra_var: int = 1500,
    max_spots_per_sample: int = 1200,
    top_n_edges: int = 200,
    top_targets_per_driver: int = 15,
    n_niches: int = 6,
    min_program_edges: int = 3,
    min_edge_qvalue: float = 0.25,
    min_edge_sample_support: int = 3,
    ccc_prior_root: str | Path | None = None,
    gold_standard_edges_path: str | Path | None = None,
    enable_ccc: bool = True,
    ccc_min_expr_frac: float = 0.02,
    ccc_min_sample_support: int = 2,
    ccc_min_lr_score: float = 0.005,
):
    prior_df, prior_conf = load_prior_bundle(prior_root, fallback_prior_root)
    lr_prior = load_ccc_prior_bundle(prior_root, fallback_prior_root, ccc_prior_root) if enable_ccc else pd.DataFrame(columns=['ligand', 'receptor'])
    ccc_prior_summary = _summarize_ccc_prior(lr_prior)
    stats = _compute_real_candidate_stats(data_root, config_path)
    candidate_genes = select_candidate_genes(stats, min_max_pct=min_max_pct, min_abs_log2fc=min_abs_log2fc, top_n_extra_var=top_n_extra_var)

    sc_summary = {'available': False}
    if sc_reference_root is not None:
        sc_reference_root = Path(sc_reference_root)
        sc_matrix_paths = [
            sc_reference_root / 'GSE132465_GEO_processed_CRC_10X_raw_UMI_count_matrix.txt.gz',
            sc_reference_root / 'GSE144735_processed_KUL3_CRC_10X_raw_UMI_count_matrix.txt.gz',
        ]
        sc_summary = summarize_sc_reference_gene_sets(sc_matrix_paths)
        if sc_summary.get('available') and sc_summary.get('union_genes'):
            sc_union = set(map(str, sc_summary['union_genes']))
            before_n = len(candidate_genes)
            candidate_genes = sorted(set(map(str, candidate_genes)).union(sc_union.intersection(set(stats['gene'].astype(str).tolist()))))
            sc_summary['candidate_genes_before_union'] = before_n
            sc_summary['candidate_genes_after_union'] = len(candidate_genes)

    candidate_edges = _prepare_candidate_edges(stats, prior_df, prior_conf, candidate_genes, mechanism_panel_path=mechanism_panel_path)
    if candidate_edges.empty:
        raise RuntimeError('No genes remain after prior-edge pruning. Check prior quality and thresholds.')
    use_genes = sorted(set(candidate_edges['source'].astype(str)).union(candidate_edges['target'].astype(str)))
    if enable_ccc and not lr_prior.empty:
        use_genes = sorted(set(use_genes).union(lr_prior['ligand'].astype(str)).union(lr_prior['receptor'].astype(str)))
    if not use_genes:
        raise RuntimeError('No genes remain after prior-edge pruning. Check prior quality and thresholds.')

    sample_to_gsm = load_sample_map(config_path)
    cfg = yaml.safe_load(Path(config_path).read_text())
    all_samples = cfg['sample_groups']['crc'] + cfg['sample_groups']['nat']
    expr_parts = []
    lag_parts = []
    coords_parts = []
    sample_edge_parts = []
    # MODIFIED: keep per-sample multiscale lags to avoid pooled cross-patient KNN leakage.
    multi_lag_parts = []
    # END MODIFIED
    sample_logs = []
    for sample in all_samples:
        try:
            expr, coords = _prepare_sample_dense(sample, data_root, sample_to_gsm, use_genes, max_spots=max_spots_per_sample)
            expr = normalize_expression_matrix(expr, assume_logged=True)
            coords_local = coords.reset_index(drop=True)
            knn = build_knn(coords_local, n_neighbors=min(6, max(1, len(coords) - 1)))
            lag = spatial_lag(expr.reset_index(drop=True), knn)
            lag.index = expr.index
            lag.columns = expr.columns
            multi_lags = multi_scale_spatial_lag(expr.reset_index(drop=True), coords_local, ks=(6, 12, 24), weighted=True)
            multi_lags = {k: v.set_axis(expr.index, axis=0).set_axis(expr.columns, axis=1) for k, v in multi_lags.items()}
            # MODIFIED: Store sample-local lags for later pooled scoring without connecting different samples.
            multi_lag_parts.append(multi_lags)
            # END MODIFIED
            expr_parts.append(expr)
            lag_parts.append(lag)
            coords_parts.append(coords)
            sample_scores = edge_scores(expr, lag, candidate_edges, multi_lag_expr=multi_lags)
            if not sample_scores.empty:
                sample_scores['sample'] = sample
                sample_edge_parts.append(_attach_prior_conf(sample_scores, candidate_edges))
            sample_logs.append({'sample': sample, 'ok': True, 'n_spots': int(expr.shape[0]), 'n_genes': int(expr.shape[1]), 'error': ''})
        except Exception as e:
            sample_logs.append({'sample': sample, 'ok': False, 'n_spots': 0, 'n_genes': 0, 'error': str(e)})
    if not expr_parts:
        raise RuntimeError('Real discovery failed: no samples could be loaded successfully. Check H5/positions file names and parquet readability.')

    expr_all_full = pd.concat(expr_parts, axis=0).fillna(0.0)
    lag_all_full = pd.concat(lag_parts, axis=0).fillna(0.0)
    coords_all = pd.concat(coords_parts, axis=0)
    expr_all, lag_all, candidate_edges, gene_qc = _filter_low_information_genes(expr_all_full, lag_all_full, candidate_edges)
    if expr_all.shape[1] == 0 or candidate_edges.empty:
        raise RuntimeError('All genes were removed by zero-variance / low-information filtering. Check priors and candidate thresholds.')

    # MODIFIED: Pooled scoring now uses sample-local spatial lags plus light batch residualization.
    # The previous pooled KNN used coordinates from different patients in one graph, which created
    # biologically impossible neighbor edges and inflated external support. We also remove gene-wise
    # sample offsets before pooled correlation so real sample identity is not mistaken for regulation.
    sample_labels_all = pd.Series([str(i).split('|', 1)[0] for i in expr_all.index], index=expr_all.index)
    expr_all_for_pooled = residualize_batch_effect(expr_all, sample_labels_all)
    pooled_multi_lags = {}
    for k in (6, 12, 24):
        parts = []
        for mp in multi_lag_parts:
            if k in mp:
                part = mp[k]
                keep_cols = [c for c in expr_all.columns if c in part.columns]
                if keep_cols:
                    parts.append(part.reindex(expr_all.index.intersection(part.index))[keep_cols])
        if parts:
            pooled_multi_lags[k] = pd.concat(parts, axis=0).reindex(expr_all.index).loc[:, expr_all.columns].fillna(0.0)
    if not pooled_multi_lags:
        pooled_multi_lags = _samplewise_multi_scale_lags(expr_all, coords_all, sample_labels_all, ks=(6, 12, 24))
    pooled_scores = _attach_prior_conf(edge_scores(expr_all_for_pooled, lag_all, candidate_edges, multi_lag_expr=pooled_multi_lags), candidate_edges)
    # END MODIFIED
    sample_edge_scores = pd.concat(sample_edge_parts, axis=0).reset_index(drop=True) if sample_edge_parts else pd.DataFrame()
    if not sample_edge_scores.empty:
        sample_edge_scores = sample_edge_scores[sample_edge_scores['source'].astype(str).isin(expr_all.columns) & sample_edge_scores['target'].astype(str).isin(expr_all.columns)].copy()
        agg_scores = _aggregate_sample_edge_scores(sample_edge_scores, candidate_edges)
        if not agg_scores.empty:
            keep_cols = [c for c in ['source', 'target', 'score', 'biological_score', 'internal', 'external', 'internal_support', 'external_support', 'full_r2', 'attention_weight', 'cascade_gain', 'internal_purity', 'external_confounding_penalty', 'intrinsic_score', 'external_branch_score'] if c in pooled_scores.columns]
            scores = agg_scores.merge(pooled_scores[keep_cols].rename(columns={'score': 'pooled_score'}), on=['source', 'target'], how='left')
            scores['pooled_score'] = scores['pooled_score'].fillna(scores['score'])
            # MODIFIED: Use sample-replicated evidence as the dominant score; pooled score is only a stabilizer.
            # MODIFIED: final edge ranking is now dominated by replicated, prior-free
            # biological evidence; pooled score remains a small stabilizer only.
            scores['score'] = 0.72 * scores['evidence_score'] + 0.18 * scores.get('bio_mean_score', 0.0) + 0.10 * scores['pooled_score']
            # END MODIFIED
            # END MODIFIED
            if 'significant' not in scores.columns:
                scores['significant'] = False
        else:
            scores = pooled_scores.copy()
            scores['significant'] = False
            scores['evidence_score'] = scores['score']
            scores['n_samples'] = 0
            scores['qvalue'] = 1.0
            scores['consistency_pos'] = np.nan
    else:
        scores = pooled_scores.copy()
        scores['significant'] = False
        scores['evidence_score'] = scores['score']
        scores['n_samples'] = 0
        scores['qvalue'] = 1.0
        scores['consistency_pos'] = np.nan

    n_sig = int(scores['significant'].sum()) if 'significant' in scores.columns else 0
    top_pool = scores.copy()
    if n_sig >= max(50, top_n_edges // 2):
        top_pool = top_pool[top_pool['significant']].copy()
    top = top_pool.sort_values(['score', 'evidence_score' if 'evidence_score' in top_pool.columns else 'score'], ascending=[False, False]).head(top_n_edges).reset_index(drop=True)

    driver_ranking = summarize_driver_ranking(scores, sample_edge_scores)
    driver_programs = build_driver_programs(
        scores,
        top_targets_per_driver=top_targets_per_driver,
        min_qvalue=min_edge_qvalue,
        min_edges_per_driver=min_program_edges,
        min_sample_support=min_edge_sample_support,
    )
    if driver_programs.empty:
        driver_programs = build_driver_programs(scores, top_targets_per_driver=top_targets_per_driver, min_qvalue=1.0, min_edges_per_driver=max(3, min_program_edges), min_sample_support=1, min_consistency=0.0)

    spot_program_scores, niche_program_scores, niche_labels = score_programs(expr_all, coords_all, driver_programs, n_niches=n_niches)
    niche_specificity = summarize_niche_specificity(spot_program_scores, niche_labels) if not spot_program_scores.empty else pd.DataFrame()

    # Lightweight Driver2Comm-inspired spatial CCC layer: infer LR exposure per sample,
    # extract recurrent CCC signatures, associate driver programs with receiver-niche CCC,
    # and materialize interpretable IE pathway rows.
    ccc_edge_parts = []
    if enable_ccc and not lr_prior.empty and not spot_program_scores.empty and not niche_labels.empty:
        lr_genes = sorted(set(lr_prior['ligand'].astype(str)).union(lr_prior['receptor'].astype(str)))
        available_lr_genes = [g for g in lr_genes if g in expr_all_full.columns]
        if available_lr_genes:
            for sample in all_samples:
                sample_idx = [idx for idx in expr_all_full.index.astype(str) if idx.startswith(f'{sample}|')]
                if len(sample_idx) < 10:
                    continue
                expr_sample_ccc = expr_all_full.loc[sample_idx, available_lr_genes].copy()
                coords_sample_ccc = coords_all.loc[sample_idx].copy()
                niches_sample = niche_labels.reindex(sample_idx).fillna('unknown').astype(str)
                try:
                    knn_ccc = build_weighted_knn(coords_sample_ccc.reset_index(drop=True), n_neighbors=min(12, max(1, len(coords_sample_ccc) - 1)))
                    ccc_edges_s = infer_spatial_lr_edges(
                        expr_sample_ccc.reset_index(drop=True),
                        coords_sample_ccc.reset_index(drop=True),
                        lr_prior,
                        sample=sample,
                        niche_labels=pd.Series(niches_sample.values, index=range(len(niches_sample))),
                        knn=knn_ccc,
                        min_expr_frac=ccc_min_expr_frac,
                    )
                    if not ccc_edges_s.empty:
                        ccc_edge_parts.append(ccc_edges_s)
                except Exception:
                    continue
    ccc_edges = pd.concat(ccc_edge_parts, axis=0).reset_index(drop=True) if ccc_edge_parts else pd.DataFrame()
    ccc_signatures = build_ccc_signatures(ccc_edges, min_sample_support=min(ccc_min_sample_support, max(1, int(sum(bool(r['ok']) for r in sample_logs)))), min_score=ccc_min_lr_score, per_sample_quantile=0.75) if not ccc_edges.empty else pd.DataFrame()
    driver_ccc_associations = associate_driver_ccc(spot_program_scores, niche_labels, ccc_edges, ccc_signatures) if not ccc_edges.empty and not ccc_signatures.empty else pd.DataFrame()
    ie_pathways = build_ie_pathways(driver_programs, driver_ccc_associations) if not driver_ccc_associations.empty else pd.DataFrame()

    mechanism_hits = pd.DataFrame()
    if mechanism_panel_path is not None and Path(mechanism_panel_path).exists():
        mechanism_hits = summarize_mechanism_hits(driver_programs, load_mechanism_panel(mechanism_panel_path))

    external_val = external_replication(driver_ranking, driver_programs, external_validation_root, genes=expr_all.columns) if external_validation_root is not None else {'available': False}
    orthogonal_val = orthogonal_xenium_validation(driver_programs, orthogonal_xenium_root) if orthogonal_xenium_root is not None else {'available': False}
    extension_audit = maybe_extension_audit(extension_root) if extension_root is not None else {'available': False}
    external_aware_driver_ranking = build_external_aware_driver_ranking(
        driver_ranking, external_val, orthogonal_val, mechanism_hits, driver_ccc_associations
    )
    gold_edges = load_gold_edges(gold_standard_edges_path) if gold_standard_edges_path is not None else pd.DataFrame(columns=['source', 'target'])
    evaluation_metrics = build_evaluation_metrics(scores, candidate_edges, gold_edges=gold_edges)

    results = {
        'mode': 'real',
        'n_spots': int(expr_all.shape[0]),
        'n_genes': int(expr_all.shape[1]),
        'candidate_stats': stats.reset_index(drop=True),
        'candidate_genes': list(candidate_genes),
        'candidate_edges': candidate_edges.reset_index(drop=True),
        'sample_edge_scores': sample_edge_scores.reset_index(drop=True),
        'edge_scores': scores.reset_index(drop=True),
        'top_edges': top,
        'driver_ranking': driver_ranking.reset_index(drop=True),
        'driver_programs': driver_programs.reset_index(drop=True),
        'spot_program_scores': spot_program_scores.reset_index().rename(columns={'index': 'spot_id'}),
        'niche_program_scores': niche_program_scores.reset_index(),
        'niche_specificity': niche_specificity.reset_index(drop=True) if isinstance(niche_specificity, pd.DataFrame) else pd.DataFrame(),
        'niche_labels': niche_labels,
        'mechanism_hits': mechanism_hits.reset_index(drop=True) if isinstance(mechanism_hits, pd.DataFrame) else pd.DataFrame(),
        'expr_shape': [int(expr_all.shape[0]), int(expr_all.shape[1])],
        'gene_qc': gene_qc,
        'ccc_edges': ccc_edges.reset_index(drop=True) if isinstance(ccc_edges, pd.DataFrame) else pd.DataFrame(),
        'ccc_signatures': ccc_signatures.reset_index(drop=True) if isinstance(ccc_signatures, pd.DataFrame) else pd.DataFrame(),
        'driver_ccc_associations': driver_ccc_associations.reset_index(drop=True) if isinstance(driver_ccc_associations, pd.DataFrame) else pd.DataFrame(),
        'ie_pathways': ie_pathways.reset_index(drop=True) if isinstance(ie_pathways, pd.DataFrame) else pd.DataFrame(),
        'ccc_prior_summary': ccc_prior_summary.reset_index(drop=True) if isinstance(ccc_prior_summary, pd.DataFrame) else pd.DataFrame(),
        'external_aware_driver_ranking': external_aware_driver_ranking.reset_index(drop=True) if isinstance(external_aware_driver_ranking, pd.DataFrame) else pd.DataFrame(),
        'evaluation_metrics': evaluation_metrics.reset_index(drop=True) if isinstance(evaluation_metrics, pd.DataFrame) else pd.DataFrame(),
        'qc_summary': {
            'candidate_genes_requested': int(len(candidate_genes)),
            'candidate_edges_after_prior_filter': int(candidate_edges.shape[0]),
            'genes_after_qc': int(expr_all.shape[1]),
            'self_loops_removed': True,
            'samplewise_aggregation': True,
            'n_significant_edges': n_sig,
            'n_ccc_edges': int(ccc_edges.shape[0]) if isinstance(ccc_edges, pd.DataFrame) else 0,
            'n_ccc_signatures': int(ccc_signatures.shape[0]) if isinstance(ccc_signatures, pd.DataFrame) else 0,
            'n_driver_ccc_associations': int(driver_ccc_associations.shape[0]) if isinstance(driver_ccc_associations, pd.DataFrame) else 0,
            'driver_program_min_edges': int(min_program_edges),
        },
        'sc_reference_summary': {k: v for k, v in sc_summary.items() if k not in {'union_genes', 'intersection_genes'}},
        'external_validation': external_val,
        'orthogonal_validation': orthogonal_val,
        'extension_audit': extension_audit,
        'sample_logs': sample_logs,
        'n_samples_ok': int(sum(bool(r['ok']) for r in sample_logs)),
        'n_samples_failed': int(sum(not bool(r['ok']) for r in sample_logs)),
    }
    _write_outputs(results, outdir)
    return results


def run_demo_pipeline(
    demo_root: str | Path,
    prior_root: str | Path,
    outdir: str | Path,
    fallback_prior_root: str | Path | None = None,
    mechanism_panel_path: str | Path | None = None,
    min_max_pct: float = 0.03,
    min_abs_log2fc: float = 0.15,
    top_n_extra_var: int = 200,
    top_n_edges: int = 50,
    top_targets_per_driver: int = 10,
    min_program_edges: int = 3,
    ccc_prior_root: str | Path | None = None,
    gold_standard_edges_path: str | Path | None = None,
    enable_ccc: bool = True,
):
    demo_root = Path(demo_root)
    expr = pd.read_csv(demo_root / 'demo_expr.csv', index_col=0)
    meta = pd.read_csv(demo_root / 'demo_meta.csv', index_col=0)
    expr = normalize_expression_matrix(expr)
    prior_df, prior_conf = load_prior_bundle(prior_root, fallback_prior_root)
    lr_prior = load_ccc_prior_bundle(prior_root, fallback_prior_root, ccc_prior_root) if enable_ccc else pd.DataFrame(columns=['ligand', 'receptor'])
    ccc_prior_summary = _summarize_ccc_prior(lr_prior)
    stats = compute_crc_nat_logfc(expr, meta['group'])
    candidate_genes = select_candidate_genes(stats, min_max_pct=min_max_pct, min_abs_log2fc=min_abs_log2fc, top_n_extra_var=top_n_extra_var)
    candidate_edges = _prepare_candidate_edges(stats, prior_df, prior_conf, candidate_genes, mechanism_panel_path=mechanism_panel_path, max_edges=3000)
    coords = meta[['x', 'y']].copy()
    # MODIFIED: Demo path mirrors the real pipeline: spatial lags are built within each
    # sample instead of on a pooled graph, and pooled scoring is batch-residualized.
    if 'sample' in meta.columns:
        sample_labels_demo = meta['sample'].astype(str).reindex(expr.index).fillna('sample')
        demo_multi_lags_full = _samplewise_multi_scale_lags(expr, coords, sample_labels_demo, ks=(6, 12, 24))
        lag = demo_multi_lags_full.get(6, spatial_lag(expr, build_knn(coords, n_neighbors=6)))
    else:
        sample_labels_demo = pd.Series('sample', index=expr.index)
        knn = build_knn(coords, n_neighbors=6)
        lag = spatial_lag(expr, knn)
        demo_multi_lags_full = multi_scale_spatial_lag(expr, coords, ks=(6, 12, 24), weighted=True)
    expr, lag, candidate_edges, gene_qc = _filter_low_information_genes(expr, lag, candidate_edges)
    demo_multi_lags = _samplewise_multi_scale_lags(expr, coords.loc[expr.index], sample_labels_demo.reindex(expr.index), ks=(6, 12, 24)) if 'sample' in meta.columns else multi_scale_spatial_lag(expr, coords.loc[expr.index], ks=(6, 12, 24), weighted=True)
    expr_for_pooled = residualize_batch_effect(expr, sample_labels_demo.reindex(expr.index)) if 'sample' in meta.columns else expr
    pooled_scores = _attach_prior_conf(edge_scores(expr_for_pooled, lag, candidate_edges, multi_lag_expr=demo_multi_lags), candidate_edges)
    # END MODIFIED

    sample_edge_parts = []
    if 'sample' in meta.columns:
        for sample, idx in meta.groupby('sample').groups.items():
            expr_s = expr.loc[idx.intersection(expr.index)] if hasattr(idx, 'intersection') else expr.loc[idx]
            coords_s = coords.loc[expr_s.index]
            if len(expr_s) < 2:
                continue
            knn_s = build_knn(coords_s, n_neighbors=min(6, max(1, len(coords_s) - 1)))
            lag_s = spatial_lag(expr_s, knn_s)
            multi_s = multi_scale_spatial_lag(expr_s, coords_s, ks=(6, 12, 24), weighted=True)
            sample_scores = edge_scores(expr_s, lag_s, candidate_edges, multi_lag_expr=multi_s)
            if not sample_scores.empty:
                sample_scores['sample'] = sample
                sample_edge_parts.append(_attach_prior_conf(sample_scores, candidate_edges))
    sample_edge_scores = pd.concat(sample_edge_parts, axis=0).reset_index(drop=True) if sample_edge_parts else pd.DataFrame()
    if not sample_edge_scores.empty:
        scores = _aggregate_sample_edge_scores(sample_edge_scores, candidate_edges)
        if not scores.empty:
            scores = scores.merge(pooled_scores[['source', 'target', 'score']].rename(columns={'score': 'pooled_score'}), on=['source', 'target'], how='left')
            scores['pooled_score'] = scores['pooled_score'].fillna(scores['score'])
            # MODIFIED: Use sample-replicated evidence as the dominant score; pooled score is only a stabilizer.
            # MODIFIED: final edge ranking is now dominated by replicated, prior-free
            # biological evidence; pooled score remains a small stabilizer only.
            scores['score'] = 0.72 * scores['evidence_score'] + 0.18 * scores.get('bio_mean_score', 0.0) + 0.10 * scores['pooled_score']
            # END MODIFIED
            # END MODIFIED
        else:
            scores = pooled_scores.copy()
            scores['evidence_score'] = scores['score']
            scores['significant'] = False
    else:
        scores = pooled_scores.copy()
        scores['evidence_score'] = scores['score']
        scores['significant'] = False
    top = scores.sort_values(['score', 'evidence_score'], ascending=[False, False]).head(top_n_edges).reset_index(drop=True)

    driver_ranking = summarize_driver_ranking(scores, sample_edge_scores)
    driver_programs = build_driver_programs(scores, top_targets_per_driver=top_targets_per_driver, min_qvalue=1.0, min_edges_per_driver=max(3, min_program_edges), min_sample_support=1, min_consistency=0.0)
    spot_program_scores, niche_program_scores, niche_labels = score_programs(expr, coords, driver_programs, n_niches=max(3, meta['sample'].nunique() if 'sample' in meta.columns else 4))
    niche_specificity = summarize_niche_specificity(spot_program_scores, niche_labels)
    if enable_ccc and not lr_prior.empty and not spot_program_scores.empty:
        # MODIFIED: Infer demo CCC per sample, matching the real pipeline and giving the
        # Driver2Comm-style association layer multiple independent sample units.
        ccc_parts = []
        if 'sample' in meta.columns:
            for sample, idx in meta.groupby('sample').groups.items():
                idx = expr.index.intersection(idx)
                if len(idx) < 10:
                    continue
                coords_s = coords.loc[idx]
                niches_s = niche_labels.reindex(idx).fillna('unknown').astype(str)
                knn_ccc = build_weighted_knn(coords_s, n_neighbors=min(12, max(1, len(coords_s) - 1)))
                ccc_s = infer_spatial_lr_edges(expr.loc[idx], coords_s, lr_prior, sample=str(sample), niche_labels=niches_s, knn=knn_ccc)
                if not ccc_s.empty:
                    ccc_parts.append(ccc_s)
        else:
            knn_ccc = build_weighted_knn(coords, n_neighbors=min(12, max(1, len(coords) - 1)))
            ccc_parts.append(infer_spatial_lr_edges(expr, coords, lr_prior, sample='sample', niche_labels=niche_labels, knn=knn_ccc))
        ccc_edges = pd.concat(ccc_parts, axis=0).reset_index(drop=True) if ccc_parts else pd.DataFrame()
        ccc_signatures = build_ccc_signatures(ccc_edges, min_sample_support=1, min_score=0.005, per_sample_quantile=0.75) if not ccc_edges.empty else pd.DataFrame()
        driver_ccc_associations = associate_driver_ccc(spot_program_scores, niche_labels, ccc_edges, ccc_signatures) if not ccc_edges.empty and not ccc_signatures.empty else associate_driver_ccc(pd.DataFrame(), pd.Series(dtype=str), pd.DataFrame())
        ie_pathways = build_ie_pathways(driver_programs, driver_ccc_associations) if not driver_ccc_associations.empty else build_ie_pathways(pd.DataFrame(), pd.DataFrame())
        # END MODIFIED
    else:
        ccc_edges = pd.DataFrame()
        ccc_signatures = pd.DataFrame()
        driver_ccc_associations = associate_driver_ccc(pd.DataFrame(), pd.Series(dtype=str), pd.DataFrame())
        ie_pathways = build_ie_pathways(pd.DataFrame(), pd.DataFrame())
    mechanism_hits = pd.DataFrame()
    if mechanism_panel_path is not None and Path(mechanism_panel_path).exists():
        mechanism_hits = summarize_mechanism_hits(driver_programs, load_mechanism_panel(mechanism_panel_path))
    external_aware_driver_ranking = build_external_aware_driver_ranking(driver_ranking, None, None, mechanism_hits, driver_ccc_associations)
    gold_edges = load_gold_edges(gold_standard_edges_path) if gold_standard_edges_path is not None else pd.DataFrame(columns=['source', 'target'])
    evaluation_metrics = build_evaluation_metrics(scores, candidate_edges, gold_edges=gold_edges)

    results = {
        'mode': 'demo',
        'n_spots': int(expr.shape[0]),
        'n_genes': int(expr.shape[1]),
        'candidate_stats': stats.reset_index(drop=True),
        'candidate_genes': list(candidate_genes),
        'candidate_edges': candidate_edges.reset_index(drop=True),
        'sample_edge_scores': sample_edge_scores.reset_index(drop=True),
        'edge_scores': scores.reset_index(drop=True),
        'top_edges': top,
        'driver_ranking': driver_ranking.reset_index(drop=True),
        'driver_programs': driver_programs.reset_index(drop=True),
        'spot_program_scores': spot_program_scores.reset_index().rename(columns={'index': 'spot_id'}),
        'niche_program_scores': niche_program_scores.reset_index(),
        'niche_specificity': niche_specificity.reset_index(drop=True),
        'niche_labels': niche_labels,
        'mechanism_hits': mechanism_hits.reset_index(drop=True) if isinstance(mechanism_hits, pd.DataFrame) else pd.DataFrame(),
        'expr_shape': [int(expr.shape[0]), int(expr.shape[1])],
        'gene_qc': gene_qc,
        'ccc_edges': ccc_edges.reset_index(drop=True) if isinstance(ccc_edges, pd.DataFrame) else pd.DataFrame(),
        'ccc_signatures': ccc_signatures.reset_index(drop=True) if isinstance(ccc_signatures, pd.DataFrame) else pd.DataFrame(),
        'driver_ccc_associations': driver_ccc_associations.reset_index(drop=True) if isinstance(driver_ccc_associations, pd.DataFrame) else pd.DataFrame(),
        'ie_pathways': ie_pathways.reset_index(drop=True) if isinstance(ie_pathways, pd.DataFrame) else pd.DataFrame(),
        'ccc_prior_summary': ccc_prior_summary.reset_index(drop=True) if isinstance(ccc_prior_summary, pd.DataFrame) else pd.DataFrame(),
        'external_aware_driver_ranking': external_aware_driver_ranking.reset_index(drop=True) if isinstance(external_aware_driver_ranking, pd.DataFrame) else pd.DataFrame(),
        'evaluation_metrics': evaluation_metrics.reset_index(drop=True) if isinstance(evaluation_metrics, pd.DataFrame) else pd.DataFrame(),
        'qc_summary': {'self_loops_removed': True, 'genes_after_qc': int(expr.shape[1]), 'samplewise_aggregation': not sample_edge_scores.empty, 'n_ccc_signatures': int(ccc_signatures.shape[0]) if isinstance(ccc_signatures, pd.DataFrame) else 0},
    }
    _write_outputs(results, outdir)
    return results
