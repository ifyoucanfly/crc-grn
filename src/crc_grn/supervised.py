from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    # MODIFIED: include the new intrinsic/extrinsic decomposition features so optional
    # supervised calibration can learn from the root-cause fixes instead of the old
    # external-dominated score alone.
    'prior', 'internal', 'external', 'score', 'raw_model_score',
    'intrinsic_score', 'external_branch_score', 'internal_purity', 'external_confounding_penalty',
    'internal_support', 'external_support', 'directional_internal_support', 'directional_corr_support',
    'internal_delta_r2', 'external_delta_r2', 'external_specificity', 'corr_xy', 'corr_yz', 'corr_xz',
    'full_r2', 'attention_weight', 'cascade_gain'
    # END MODIFIED
]


@dataclass
class CalibrationResult:
    available: bool
    reason: str
    metrics: pd.DataFrame
    calibrated_edges: pd.DataFrame


def _edge_key(df: pd.DataFrame) -> pd.Series:
    return df['source'].astype(str).str.upper() + '->' + df['target'].astype(str).str.upper()


def calibrate_edge_scores_with_gold(
    edge_scores: pd.DataFrame,
    gold_edges: pd.DataFrame | None,
    feature_columns: Iterable[str] = FEATURE_COLUMNS,
    max_epochs: int = 200,
    patience: int = 20,
    seed: int = 42,
) -> CalibrationResult:
    """Optional supervised calibration layer for datasets with held-out gold GRN labels.

    The package is primarily unsupervised/knowledge-guided. When users provide an
    independent gold-standard edge list, this function trains a tiny MLP with weighted
    BCE, AdamW, learning-rate scheduling, gradient clipping, early stopping and AMP on
    CUDA. It is deliberately optional so the core GRN pipeline remains runnable without
    labels.
    """
    if edge_scores is None or edge_scores.empty:
        return CalibrationResult(False, 'empty_edge_scores', pd.DataFrame(), pd.DataFrame())
    if gold_edges is None or gold_edges.empty:
        return CalibrationResult(False, 'no_gold_edges_supplied', pd.DataFrame(), edge_scores.copy())
    try:
        import torch
        from torch import nn
        from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
        from sklearn.model_selection import GroupShuffleSplit
        from sklearn.preprocessing import StandardScaler
    except Exception as e:
        return CalibrationResult(False, f'missing_dependency:{type(e).__name__}', pd.DataFrame(), edge_scores.copy())

    df = edge_scores.copy()
    df['_edge'] = _edge_key(df)
    gold = set(_edge_key(gold_edges))
    df['label'] = df['_edge'].isin(gold).astype(int)
    if df['label'].nunique() < 2:
        return CalibrationResult(False, 'single_class_labels', pd.DataFrame(), df.drop(columns=['_edge']))
    feat_cols = [c for c in feature_columns if c in df.columns]
    if len(feat_cols) < 4:
        return CalibrationResult(False, 'too_few_feature_columns', pd.DataFrame(), df.drop(columns=['_edge']))
    X = df[feat_cols].apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan).fillna(0.0).values.astype('float32')
    y = df['label'].values.astype('float32')
    groups = df['target'].astype(str).values if 'target' in df.columns else np.arange(len(df))
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    train_idx, val_idx = next(splitter.split(X, y, groups=groups))
    scaler = StandardScaler().fit(X[train_idx])
    X = scaler.transform(X).astype('float32')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(seed)
    model = nn.Sequential(
        nn.Linear(X.shape[1], 32), nn.LayerNorm(32), nn.GELU(), nn.Dropout(0.15),
        nn.Linear(32, 16), nn.GELU(), nn.Dropout(0.10), nn.Linear(16, 1)
    ).to(device)
    pos_weight = torch.tensor([(len(train_idx) - y[train_idx].sum()) / max(y[train_idx].sum(), 1.0)], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=5)
    scaler_amp = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))
    Xtr = torch.tensor(X[train_idx], device=device)
    ytr = torch.tensor(y[train_idx, None], device=device)
    Xva = torch.tensor(X[val_idx], device=device)
    yva_np = y[val_idx]
    best_state = None
    best_ap = -np.inf
    bad = 0
    for _ in range(int(max_epochs)):
        model.train()
        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
            loss = loss_fn(model(Xtr), ytr)
        scaler_amp.scale(loss).backward()
        scaler_amp.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler_amp.step(opt)
        scaler_amp.update()
        model.eval()
        with torch.no_grad():
            pred = torch.sigmoid(model(Xva)).detach().cpu().numpy().ravel()
        ap = average_precision_score(yva_np, pred) if len(np.unique(yva_np)) == 2 else 0.0
        scheduler.step(ap)
        if ap > best_ap + 1e-5:
            best_ap = ap
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if bad >= int(patience):
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred_all = torch.sigmoid(model(torch.tensor(X, device=device))).detach().cpu().numpy().ravel()
        pred_val = pred_all[val_idx]
    df['calibrated_score'] = pred_all
    metrics = pd.DataFrame([
        {'metric': 'calibration_val_AUROC', 'value': float(roc_auc_score(yva_np, pred_val)) if len(np.unique(yva_np)) == 2 else np.nan},
        {'metric': 'calibration_val_AUPRC', 'value': float(average_precision_score(yva_np, pred_val)) if len(np.unique(yva_np)) == 2 else np.nan},
        {'metric': 'calibration_val_F1_at_0.5', 'value': float(f1_score(yva_np, pred_val >= 0.5)) if len(np.unique(yva_np)) == 2 else np.nan},
    ])
    return CalibrationResult(True, 'ok', metrics, df.drop(columns=['_edge']))
