#!/usr/bin/env python3
"""Audit a crc_grn real-run output directory and write concise diagnostics.

Usage:
  python scripts/analyze_real_output.py --results results/real_run --out results/real_run_diagnostics.csv
"""
# MODIFIED: New output-analysis helper added after auditing real_run(1).zip. It checks
# statistical calibration, prior-source counting, Driver2Comm association density, and
# validation readiness so future runs can be judged before manual biological inspection.
import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _metric(rows: list[dict[str, Any]], section: str, name: str, value: Any, status: str = "info", note: str = "") -> None:
    rows.append({"section": section, "metric": name, "value": value, "status": status, "note": note})


def _safe_float(x: Any) -> float:
    try:
        if pd.isna(x):
            return float("nan")
        return float(x)
    except Exception:
        return float("nan")


def analyze(results_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    run_summary = _read_csv(results_dir / "run_summary.csv")
    edge_scores = _read_csv(results_dir / "edge_scores.csv")
    metrics = _read_csv(results_dir / "evaluation_metrics.csv")
    drivers = _read_csv(results_dir / "driver_ranking.csv")
    ccc_assoc = _read_csv(results_dir / "driver_ccc_associations.csv")
    ccc_sig = _read_csv(results_dir / "ccc_signatures.csv")
    ext_summary = _read_csv(results_dir / "validation" / "external_summary.csv")
    ortho_status = _read_csv(results_dir / "validation" / "orthogonal_run_level_status.csv")

    if not run_summary.empty:
        r = run_summary.iloc[0]
        for col in ["n_spots", "n_genes", "n_candidate_genes", "n_candidate_edges", "n_significant_edges", "n_driver_programs", "n_ccc_signatures", "n_driver_ccc_associations", "n_samples_ok", "n_samples_failed"]:
            if col in run_summary.columns:
                _metric(rows, "run_summary", col, r[col])

    if not edge_scores.empty:
        n_edges = len(edge_scores)
        n_sig = int(edge_scores.get("significant", pd.Series(False, index=edge_scores.index)).fillna(False).sum())
        sig_rate = n_sig / max(n_edges, 1)
        status = "warn" if sig_rate > 0.25 else "ok"
        _metric(rows, "edge_calibration", "significant_edge_rate", round(sig_rate, 4), status, "High rates usually indicate q-value inflation or overly broad priors.")
        for col in ["qvalue", "qvalue_effect", "pvalue_effect", "score", "evidence_score"]:
            if col in edge_scores.columns:
                s = pd.to_numeric(edge_scores[col], errors="coerce")
                _metric(rows, "edge_calibration", f"{col}_median", round(float(s.median()), 6) if s.notna().any() else np.nan)
                _metric(rows, "edge_calibration", f"{col}_q95", round(float(s.quantile(0.95)), 6) if s.notna().any() else np.nan)
        if {"prior_sources", "prior_n_sources"}.issubset(edge_scores.columns):
            multi_source_bug = edge_scores[edge_scores["prior_sources"].astype(str).str.contains(";", regex=False, na=False) & (pd.to_numeric(edge_scores["prior_n_sources"], errors="coerce") <= 1)]
            _metric(rows, "prior", "semicolon_sources_counted_as_one", len(multi_source_bug), "warn" if len(multi_source_bug) else "ok", "Should be zero after v3 load_prior_bundle fix.")
        if {"mean_external", "mean_internal"}.issubset(drivers.columns):
            top = drivers.head(20).copy()
            ext_dom = (pd.to_numeric(top["mean_external"], errors="coerce").abs() > pd.to_numeric(top["mean_internal"], errors="coerce").abs()).mean()
            _metric(rows, "driver_ranking", "top20_external_dominance_fraction", round(float(ext_dom), 4), "warn" if ext_dom > 0.5 else "ok", "High values suggest external branch leakage into intrinsic GRN ranking.")

    if not metrics.empty:
        filt = metrics[(metrics.get("benchmark", "") == "prior_overlap_trrust") & (metrics.get("benchmark_type", "") == "internal_silver")]
        for _, m in filt.iterrows():
            metric = str(m.get("metric", ""))
            score_col = str(m.get("score_col", ""))
            value = _safe_float(m.get("value"))
            status = "warn" if metric.lower() in {"roc_auc", "auroc", "average_precision", "auprc"} and np.isfinite(value) and value < 0.55 else "info"
            _metric(rows, "evaluation", f"trrust_{score_col}_{metric}", round(value, 6) if np.isfinite(value) else value, status)

    if not ccc_assoc.empty:
        n_assoc = len(ccc_assoc)
        n_sig_assoc = int(ccc_assoc.get("significant", pd.Series(False, index=ccc_assoc.index)).fillna(False).sum()) if "significant" in ccc_assoc else 0
        _metric(rows, "driver2comm", "n_driver_ccc_associations", n_assoc)
        _metric(rows, "driver2comm", "n_significant_driver_ccc_associations", n_sig_assoc, "warn" if n_sig_assoc > 500 else "ok", "Thousands of significant associations usually indicate sample-effect confounding.")
        if {"driver", "signature_id"}.issubset(ccc_assoc.columns):
            density = len(ccc_assoc) / max(ccc_assoc["driver"].nunique() * ccc_assoc["signature_id"].nunique(), 1)
            _metric(rows, "driver2comm", "driver_signature_grid_density", round(float(density), 4), "warn" if density > 0.95 else "ok", "A full grid is expected before filtering, but significant links should be sparse.")
        for col in ["rho", "sample_rho", "pseudo_spearman_rho", "niche_residual_rho"]:
            if col in ccc_assoc.columns:
                s = pd.to_numeric(ccc_assoc[col], errors="coerce")
                _metric(rows, "driver2comm", f"{col}_median", round(float(s.median()), 6) if s.notna().any() else np.nan)
    if not ccc_sig.empty:
        _metric(rows, "ccc_signatures", "n_signatures", len(ccc_sig))
        if "support_rate" in ccc_sig:
            _metric(rows, "ccc_signatures", "support_rate_median", round(float(pd.to_numeric(ccc_sig["support_rate"], errors="coerce").median()), 6))
        if "crc_specificity" in ccc_sig:
            _metric(rows, "ccc_signatures", "crc_specificity_available", True, "ok")
        else:
            _metric(rows, "ccc_signatures", "crc_specificity_available", False, "warn", "v3 adds CRC/NAT-aware CCC signature metadata.")

    if not ext_summary.empty:
        r = ext_summary.iloc[0]
        for col in ["core_support_rate", "spearman_rho_all", "spearman_pvalue_all", "coverage_pass_rate"]:
            if col in ext_summary.columns:
                val = _safe_float(r[col])
                status = "warn" if col == "spearman_rho_all" and np.isfinite(val) and val < 0 else "info"
                _metric(rows, "external_validation", col, round(val, 6) if np.isfinite(val) else val, status)
    if not ortho_status.empty:
        r = ortho_status.iloc[0]
        loaded = int(r.get("n_loaded_runs", 0)) if pd.notna(r.get("n_loaded_runs", np.nan)) else 0
        ready = bool(r.get("three_run_validation_ready", False))
        _metric(rows, "orthogonal_validation", "n_loaded_runs", loaded, "warn" if loaded < 3 else "ok")
        _metric(rows, "orthogonal_validation", "three_run_validation_ready", ready, "ok" if ready else "warn", "Stage report requires treating GSE280314 as orthogonal high-resolution validation, not full external replication.")

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True, help="Path to a crc_grn results directory, e.g. results/real_run")
    parser.add_argument("--out", type=Path, default=None, help="Output CSV path")
    args = parser.parse_args()
    df = analyze(args.results)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
# END MODIFIED
