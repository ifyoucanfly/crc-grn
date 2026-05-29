#!/usr/bin/env python
"""Run the CRC GRN/Driver2Comm pipeline with the validation upgrades enabled.

Example:
python scripts/run_real_validation_upgrade.py \
  --data-root data/main_visiumhd \
  --config config/samples.yaml \
  --prior-root resources/grn_prior \
  --ccc-prior-root resources/ccc_prior \
  --sc-reference-root data/sc_reference \
  --orthogonal-xenium-root data/orthogonal_xenium_gse280314 \
  --external-validation-root data/external_validation_gse226997 \
  --gold-standard-edges resources/gold_standard/collectri_crc_holdout.tsv \
  --outdir results/real_run_validation_upgrade
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / 'src'))
from crc_grn.pipeline import run_real_pipeline


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description='Run CRC GRN with robust validation upgrades.')
    ap.add_argument('--data-root', type=Path, default=root / 'data' / 'main_visiumhd')
    ap.add_argument('--config', type=Path, default=root / 'config' / 'samples.yaml')
    ap.add_argument('--prior-root', type=Path, default=root / 'resources' / 'grn_prior')
    ap.add_argument('--fallback-prior-root', type=Path, default=root / 'resources' / 'demo_prior')
    ap.add_argument('--ccc-prior-root', type=Path, default=root / 'resources' / 'ccc_prior')
    ap.add_argument('--sc-reference-root', type=Path, default=root / 'data' / 'sc_reference')
    ap.add_argument('--orthogonal-xenium-root', type=Path, default=root / 'data' / 'orthogonal_xenium_gse280314')
    ap.add_argument('--external-validation-root', type=Path, default=root / 'data' / 'external_validation_gse226997')
    ap.add_argument('--extension-root', type=Path, default=root / 'data' / 'extension_gse267401')
    ap.add_argument('--mechanism-panel', type=Path, default=root / 'config' / 'mechanism_panel.yaml')
    ap.add_argument('--gold-standard-edges', type=Path, default=None, help='Optional independent TF-target edge list for AUROC/AUPRC/F1.')
    ap.add_argument('--outdir', type=Path, default=root / 'results' / 'real_run_validation_upgrade')
    ap.add_argument('--max-spots-per-sample', type=int, default=1200)
    args = ap.parse_args()

    res = run_real_pipeline(
        data_root=args.data_root,
        config_path=args.config,
        prior_root=args.prior_root,
        fallback_prior_root=args.fallback_prior_root,
        ccc_prior_root=args.ccc_prior_root,
        sc_reference_root=args.sc_reference_root,
        orthogonal_xenium_root=args.orthogonal_xenium_root,
        external_validation_root=args.external_validation_root,
        extension_root=args.extension_root,
        mechanism_panel_path=args.mechanism_panel,
        gold_standard_edges_path=args.gold_standard_edges,
        outdir=args.outdir,
        max_spots_per_sample=args.max_spots_per_sample,
    )
    print(f"Outputs saved in: {args.outdir}")
    print(res['run_summary'].to_string(index=False) if 'run_summary' in res else res['top_edges'].head(5).to_string(index=False))


if __name__ == '__main__':
    main()
