from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def _edge_key(df: pd.DataFrame, s_col: str = 'source', t_col: str = 'target') -> pd.Series:
    return df[s_col].astype(str).str.upper() + '->' + df[t_col].astype(str).str.upper()


def load_gold_edges(path: str | Path | None) -> pd.DataFrame:
    """Load an independent TF-target gold/silver standard edge list.

    Accepted column aliases: source/tf/regulator/source_gene and target/target_gene.
    This function is intentionally strict: if source/target columns cannot be inferred,
    an empty frame is returned so the pipeline does not silently fabricate labels.
    """
    if path is None:
        return pd.DataFrame(columns=['source', 'target'])
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=['source', 'target'])
    sep = ',' if path.suffix.lower() == '.csv' else '\t'
    df = pd.read_csv(path, sep=sep, low_memory=False)
    rename = {}
    for c in df.columns:
        cl = str(c).strip().lower()
        if cl in {'tf', 'regulator', 'source_gene', 'source', 'from'}:
            rename[c] = 'source'
        if cl in {'target_gene', 'target', 'to'}:
            rename[c] = 'target'
    df = df.rename(columns=rename)
    if not {'source', 'target'}.issubset(df.columns):
        return pd.DataFrame(columns=['source', 'target'])
    out = df[['source', 'target']].dropna().copy()
    out['source'] = out['source'].astype(str).str.upper()
    out['target'] = out['target'].astype(str).str.upper()
    out = out[out['source'] != out['target']]
    return out.drop_duplicates().reset_index(drop=True)


def _best_binary_metrics(y: np.ndarray, scores: np.ndarray) -> dict:
    from sklearn.metrics import precision_recall_curve
    precision, recall, thr = precision_recall_curve(y, scores)
    f1 = (2 * precision * recall) / np.maximum(precision + recall, 1e-12)
    best_i = int(np.nanargmax(f1))
    best_thr = float(thr[best_i]) if best_i < len(thr) else float(np.nanmax(scores))
    return {
        'F1_best': float(f1[best_i]),
        'precision_at_best_f1': float(precision[best_i]),
        'recall_at_best_f1': float(recall[best_i]),
        'best_threshold': best_thr,
    }


def edge_prediction_metrics(
    pred_edges: pd.DataFrame,
    gold_edges: pd.DataFrame | None = None,
    score_col: str = 'score',
    benchmark_name: str = 'independent_gold',
    benchmark_type: str = 'external_gold',
) -> pd.DataFrame:
    """Compute AUROC/AUPRC/F1/precision@K for a supplied gold-standard edge set.

    Metrics are only reported when both classes are present among the scored edges.
    The returned table explicitly labels the benchmark type so internal sanity checks
    cannot be mistaken for independent validation.
    """
    cols = ['benchmark', 'benchmark_type', 'score_col', 'metric', 'value', 'n_edges', 'n_positive', 'threshold', 'note']
    if pred_edges is None or pred_edges.empty or gold_edges is None or gold_edges.empty or score_col not in pred_edges.columns:
        return pd.DataFrame(columns=cols)
    df = pred_edges.copy()
    if not {'source', 'target'}.issubset(df.columns):
        return pd.DataFrame(columns=cols)
    df['_edge'] = _edge_key(df)
    gold = set(_edge_key(gold_edges))
    df['label'] = df['_edge'].isin(gold).astype(int)
    n_pos = int(df['label'].sum())
    n_edges = int(len(df))
    if n_pos == 0 or n_pos == n_edges:
        return pd.DataFrame([{
            'benchmark': benchmark_name,
            'benchmark_type': benchmark_type,
            'score_col': score_col,
            'metric': 'not_applicable_single_class',
            'value': np.nan,
            'n_edges': n_edges,
            'n_positive': n_pos,
            'threshold': np.nan,
            'note': 'Need at least one positive and one negative edge in the scored candidate universe.',
        }], columns=cols)
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score
        scores = pd.to_numeric(df[score_col], errors='coerce').replace([np.inf, -np.inf], np.nan).fillna(0.0).values
        y = df['label'].values.astype(int)
        rows = [
            {'metric': 'AUROC', 'value': float(roc_auc_score(y, scores)), 'threshold': np.nan},
            {'metric': 'AUPRC', 'value': float(average_precision_score(y, scores)), 'threshold': np.nan},
        ]
        best = _best_binary_metrics(y, scores)
        rows.extend([
            {'metric': 'F1_best', 'value': best['F1_best'], 'threshold': best['best_threshold']},
            {'metric': 'precision_at_best_f1', 'value': best['precision_at_best_f1'], 'threshold': best['best_threshold']},
            {'metric': 'recall_at_best_f1', 'value': best['recall_at_best_f1'], 'threshold': best['best_threshold']},
        ])
        for k in [10, 20, 50, 100]:
            kk = min(k, n_edges)
            if kk > 0:
                top = df.assign(_score=scores).sort_values('_score', ascending=False).head(kk)
                rows.append({'metric': f'precision_at_{k}', 'value': float(top['label'].mean()), 'threshold': np.nan})
        out = pd.DataFrame(rows)
        out['benchmark'] = benchmark_name
        out['benchmark_type'] = benchmark_type
        out['score_col'] = score_col
        out['n_edges'] = n_edges
        out['n_positive'] = n_pos
        out['note'] = '' if benchmark_type == 'external_gold' else 'Internal/silver sanity check; do not report as independent performance.'
        return out[cols]
    except Exception as e:
        return pd.DataFrame([{
            'benchmark': benchmark_name,
            'benchmark_type': benchmark_type,
            'score_col': score_col,
            'metric': f'failed:{type(e).__name__}',
            'value': np.nan,
            'n_edges': n_edges,
            'n_positive': n_pos,
            'threshold': np.nan,
            'note': str(e),
        }], columns=cols)


def _prior_source_gold(candidate_edges: pd.DataFrame, source_name: str) -> pd.DataFrame:
    if candidate_edges is None or candidate_edges.empty or 'prior_sources' not in candidate_edges.columns:
        return pd.DataFrame(columns=['source', 'target'])
    df = candidate_edges.copy()
    mask = df['prior_sources'].fillna('').astype(str).str.lower().str.split(';').apply(lambda xs: source_name.lower() in set(xs))
    return df.loc[mask, ['source', 'target']].drop_duplicates().reset_index(drop=True)


def build_evaluation_metrics(
    edge_scores: pd.DataFrame,
    candidate_edges: pd.DataFrame | None = None,
    gold_edges: pd.DataFrame | None = None,
    score_cols: Iterable[str] = ('score', 'evidence_score', 'calibrated_score'),
) -> pd.DataFrame:
    """Build a non-empty evaluation report when possible.

    Priority 1 is an independent gold-standard file supplied by the user. When it is
    absent, the function additionally emits clearly-labelled prior-overlap sanity checks
    against CollecTRI/TRRUST provenance in the candidate edge table. These sanity checks
    prevent blank reports during development but are not presented as performance claims.
    """
    cols = ['benchmark', 'benchmark_type', 'score_col', 'metric', 'value', 'n_edges', 'n_positive', 'threshold', 'note']
    if edge_scores is None or edge_scores.empty:
        return pd.DataFrame([{
            'benchmark': 'none', 'benchmark_type': 'none', 'score_col': '', 'metric': 'empty_edge_scores',
            'value': np.nan, 'n_edges': 0, 'n_positive': 0, 'threshold': np.nan, 'note': 'No edge scores available.'
        }], columns=cols)
    frames = []
    for score_col in score_cols:
        if score_col not in edge_scores.columns:
            continue
        if gold_edges is not None and not gold_edges.empty:
            frames.append(edge_prediction_metrics(edge_scores, gold_edges, score_col, 'user_supplied_gold', 'external_gold'))
        if candidate_edges is not None and not candidate_edges.empty and 'prior_sources' in candidate_edges.columns:
            sources = sorted(set(';'.join(candidate_edges['prior_sources'].dropna().astype(str)).lower().split(';')) - {''})
            for src in sources:
                src_gold = _prior_source_gold(candidate_edges, src)
                # Require a minimally useful label distribution.
                if src_gold.empty:
                    continue
                frames.append(edge_prediction_metrics(edge_scores, src_gold, score_col, f'prior_overlap_{src}', 'internal_silver'))
    out = pd.concat([f for f in frames if f is not None and not f.empty], ignore_index=True) if frames else pd.DataFrame(columns=cols)
    if out.empty:
        return pd.DataFrame([{
            'benchmark': 'none', 'benchmark_type': 'none', 'score_col': '', 'metric': 'no_usable_gold_or_silver_labels',
            'value': np.nan, 'n_edges': int(len(edge_scores)), 'n_positive': 0, 'threshold': np.nan,
            'note': 'Provide --gold-standard-edges for AUROC/AUPRC/F1, or ensure prior_sources are available for internal sanity checks.'
        }], columns=cols)
    return out[cols]
