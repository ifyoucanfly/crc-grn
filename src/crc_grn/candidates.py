from __future__ import annotations

import numpy as np
import pandas as pd


def compute_crc_nat_logfc(expr: pd.DataFrame, groups: pd.Series, crc_label: str = 'CRC', nat_label: str = 'NAT', pseudocount: float = 1.0) -> pd.DataFrame:
    crc = expr.loc[groups == crc_label]
    nat = expr.loc[groups == nat_label]
    crc_mean = crc.mean(axis=0)
    nat_mean = nat.mean(axis=0)
    log2fc = np.log2((crc_mean + pseudocount) / (nat_mean + pseudocount))
    pct_crc = (crc > 0).mean(axis=0)
    pct_nat = (nat > 0).mean(axis=0)
    out = pd.DataFrame({'gene': expr.columns, 'log2fc': log2fc.values, 'pct_crc': pct_crc.values, 'pct_nat': pct_nat.values})
    out['max_pct'] = out[['pct_crc', 'pct_nat']].max(axis=1)
    out['abs_log2fc'] = out['log2fc'].abs()
    return out.sort_values('abs_log2fc', ascending=False).reset_index(drop=True)


def select_candidate_genes(stats_df: pd.DataFrame, min_max_pct: float = 0.03, min_abs_log2fc: float = 0.15, top_n_extra_var: int = 1200) -> list[str]:
    base = stats_df[(stats_df['max_pct'] >= min_max_pct) & (stats_df['abs_log2fc'] >= min_abs_log2fc)]['gene'].astype(str).tolist()
    extras = stats_df.sort_values(['abs_log2fc', 'max_pct'], ascending=[False, False])['gene'].astype(str).head(top_n_extra_var).tolist()
    return sorted(set(base).union(extras))


def shrink_prior_edges(prior_df: pd.DataFrame, candidate_genes: list[str], allow_self_loops: bool = False) -> pd.DataFrame:
    gene_set = set(map(str, candidate_genes))
    out = prior_df[prior_df['source'].astype(str).isin(gene_set) & prior_df['target'].astype(str).isin(gene_set)].copy()
    if not allow_self_loops:
        out = out[out['source'].astype(str) != out['target'].astype(str)].copy()
    return out.reset_index(drop=True)
