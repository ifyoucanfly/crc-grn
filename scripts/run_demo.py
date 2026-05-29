#!/usr/bin/env python
from pathlib import Path
import sys
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / 'src'))
from crc_grn.pipeline import run_demo_pipeline


def _ensure_demo_bundle(root: Path) -> None:
    """Create a small deterministic demo dataset when the real demo bundle is absent."""
    data_dir = root / 'data' / 'demo_bundle'
    expr_path = data_dir / 'demo_expr.csv'
    meta_path = data_dir / 'demo_meta.csv'
    if expr_path.exists() and meta_path.exists():
        return
    data_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    genes = [
        'STAT3', 'CXCL8', 'MMP9', 'IL6', 'MYC', 'MKI67', 'CD44', 'JUN', 'VIM', 'COL1A1',
        'TGFB1', 'TGFBR1', 'TGFBR2', 'SPP1', 'ITGA4', 'ITGB1', 'CXCL10', 'CXCR3',
        'CXCL12', 'CXCR4', 'VEGFA', 'KDR', 'FLT1', 'MIF', 'CD74', 'JAG1', 'NOTCH1',
        'HGF', 'MET', 'APOE', 'LRP1', 'GAPDH', 'ACTB'
    ]
    n_per_sample = 70
    samples = ['CRC_A', 'CRC_B', 'NAT_A', 'NAT_B']
    rows = []
    meta = []
    for si, sample in enumerate(samples):
        is_crc = sample.startswith('CRC')
        base_x = (si % 2) * 120
        base_y = (si // 2) * 120
        for i in range(n_per_sample):
            x = base_x + rng.normal(50, 18)
            y = base_y + rng.normal(50, 18)
            vals = rng.poisson(1.5, size=len(genes)).astype(float)
            g = {gene: j for j, gene in enumerate(genes)}
            # Embed a weak but recoverable CRC intrinsic/extrinsic program.
            if is_crc:
                vals[g['STAT3']] += rng.poisson(4)
                vals[g['IL6']] += rng.poisson(3)
                vals[g['CXCL8']] += rng.poisson(3)
                vals[g['MMP9']] += rng.poisson(2)
                vals[g['MYC']] += rng.poisson(2)
                vals[g['MKI67']] += rng.poisson(3)
                vals[g['SPP1']] += rng.poisson(2)
                vals[g['CD44']] += rng.poisson(2)
                if x > base_x + 50:
                    vals[g['HGF']] += rng.poisson(3)
                    vals[g['MET']] += rng.poisson(2)
                if y > base_y + 50:
                    vals[g['CXCL10']] += rng.poisson(3)
                    vals[g['CXCR3']] += rng.poisson(2)
            else:
                vals[g['JUN']] += rng.poisson(1)
                vals[g['VIM']] += rng.poisson(1)
            spot = f'{sample}_{i:03d}'
            rows.append(vals)
            meta.append({'spot': spot, 'sample': sample, 'group': 'CRC' if is_crc else 'NAT', 'x': x, 'y': y})
    expr = pd.DataFrame(rows, index=[m['spot'] for m in meta], columns=genes)
    meta_df = pd.DataFrame(meta).set_index('spot')
    expr.to_csv(expr_path)
    meta_df.to_csv(meta_path)


root = Path(__file__).resolve().parents[1]
_ensure_demo_bundle(root)
results = run_demo_pipeline(
    demo_root=root / 'data' / 'demo_bundle',
    prior_root=root / 'resources' / 'grn_prior',
    fallback_prior_root=root / 'resources' / 'demo_prior',
    ccc_prior_root=root / 'resources' / 'ccc_prior',
    outdir=root / 'results' / 'demo_run',
)
print(results['top_edges'].head(10).to_string(index=False))
print(f"\nOutputs saved in: {root / 'results' / 'demo_run'}")
