from __future__ import annotations

from pathlib import Path
import yaml
import pandas as pd


def load_mechanism_panel(path: str | Path) -> dict[str, list[str]]:
    with open(path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    return {k: [str(x).upper() for x in v] for k, v in cfg.get('mechanism_axes', {}).items()}


def summarize_mechanism_hits(driver_programs: pd.DataFrame, panel: dict[str, list[str]]) -> pd.DataFrame:
    rows = []
    if driver_programs.empty:
        return pd.DataFrame(columns=['driver', 'mechanism_axis', 'n_overlap', 'overlap_genes'])
    for driver, df in driver_programs.groupby('driver'):
        gene_set = set(df['target'].astype(str).str.upper()) | {str(driver).upper()}
        for axis, genes in panel.items():
            overlap = sorted(gene_set.intersection(set(genes)))
            if overlap:
                rows.append({
                    'driver': driver,
                    'mechanism_axis': axis,
                    'n_overlap': len(overlap),
                    'overlap_genes': ';'.join(overlap),
                })
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=['driver', 'mechanism_axis', 'n_overlap', 'overlap_genes'])
    return out.sort_values(['n_overlap', 'driver', 'mechanism_axis'], ascending=[False, True, True]).reset_index(drop=True)
