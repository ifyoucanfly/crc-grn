from __future__ import annotations

from pathlib import Path
import pandas as pd


def _safe_read_tsv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, sep='\t', low_memory=False)


def load_collectri(path: str | Path) -> pd.DataFrame:
    df = _safe_read_tsv(path)
    rename = {
        'source_genesymbol': 'source',
        'target_genesymbol': 'target',
        'consensus_stimulation': 'stim',
        'consensus_inhibition': 'inh',
        'references': 'references',
        'sources': 'sources',
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    keep = [c for c in ['source', 'target', 'stim', 'inh', 'references', 'sources'] if c in df.columns]
    df = df[keep].copy()
    df['prior_source'] = 'collectri'
    return df


def load_trrust(path: str | Path) -> pd.DataFrame:
    df = _safe_read_tsv(path)
    rename = {
        'source_genesymbol': 'source',
        'target_genesymbol': 'target',
        'references': 'references',
        'sources': 'sources',
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    keep = [c for c in ['source', 'target', 'references', 'sources'] if c in df.columns]
    df = df[keep].copy()
    df['prior_source'] = 'trrust'
    return df


def load_omnipath_signaling(path: str | Path) -> pd.DataFrame:
    df = _safe_read_tsv(path)
    rename = {'source_genesymbol': 'source', 'target_genesymbol': 'target'}
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    keep = [c for c in ['source', 'target', 'consensus_stimulation', 'consensus_inhibition', 'references', 'sources'] if c in df.columns]
    return df[keep].copy()


def load_intercell(path: str | Path) -> pd.DataFrame:
    return _safe_read_tsv(path)


def _infer_regulatory_sign(row: pd.Series) -> int:
    """Infer activation/repression direction from prior annotations.

    Returns +1 for activating/stimulatory evidence, -1 for inhibitory evidence and 0
    when the direction is unknown or conflicting. Direction-aware scoring is critical
    for GRN recovery because many TF-target edges in CollecTRI/TRRUST are repressive;
    treating every biologically valid edge as a positive correlation systematically
    suppresses those signals.
    """
    stim = str(row.get('stim', row.get('consensus_stimulation', ''))).lower()
    inh = str(row.get('inh', row.get('consensus_inhibition', ''))).lower()
    mode = str(row.get('mode', row.get('regulation', ''))).lower()
    pos_tokens = {'true', '1', 'yes', 'activation', 'activator', 'stimulation', 'up'}
    neg_tokens = {'true', '1', 'yes', 'repression', 'repressor', 'inhibition', 'down'}
    is_pos = any(t in stim for t in pos_tokens) or any(t in mode for t in {'activation', 'stimulation', 'activator', 'up'})
    is_neg = any(t in inh for t in neg_tokens) or any(t in mode for t in {'repression', 'inhibition', 'repressor', 'down'})
    if is_pos and not is_neg:
        return 1
    if is_neg and not is_pos:
        return -1
    return 0


def merge_transcriptional_priors(collectri_df: pd.DataFrame, trrust_df: pd.DataFrame) -> pd.DataFrame:
    common = sorted(set(collectri_df.columns).union(trrust_df.columns))
    frames = []
    for src_df in (collectri_df, trrust_df):
        df = src_df.copy()
        for c in common:
            if c not in df.columns:
                df[c] = pd.NA
        frames.append(df[common])
    merged = pd.concat(frames, ignore_index=True)
    merged = merged.drop_duplicates(subset=['source', 'target', 'prior_source'])
    merged['regulatory_sign_raw'] = merged.apply(_infer_regulatory_sign, axis=1)

    def _merge_sign(values) -> int:
        vals = [int(v) for v in values if pd.notna(v) and int(v) != 0]
        if not vals:
            return 0
        pos = sum(v > 0 for v in vals)
        neg = sum(v < 0 for v in vals)
        if pos > neg:
            return 1
        if neg > pos:
            return -1
        return 0

    agg = merged.groupby(['source', 'target'], as_index=False).agg(
        prior_source=('prior_source', lambda s: ';'.join(sorted(set(map(str, s))))),
        references=('references', lambda s: ';'.join(sorted(set(str(x) for x in s if pd.notna(x) and str(x) not in {'', 'nan'}))) if 'references' in merged.columns else ''),
        sources=('sources', lambda s: ';'.join(sorted(set(str(x) for x in s if pd.notna(x) and str(x) not in {'', 'nan'}))) if 'sources' in merged.columns else ''),
        regulatory_sign=('regulatory_sign_raw', _merge_sign),
    )
    return agg


def _first_present(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lookup = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        if c in df.columns:
            return c
        if c.lower() in lookup:
            return lookup[c.lower()]
    return None


def _split_complex_symbol(value: object) -> list[str]:
    """Return candidate gene symbols from simple or complex LR database entries.

    Many CCC resources store complexes as strings such as ``A_B`` or ``A:B``.
    For this lightweight spatial layer we expand them into subunit-level candidate
    pairs. This is conservative for discovery because true complex-aware scoring is
    handled upstream by CellPhoneDB/CellChat, while this package only needs a broad
    LR prior search space.
    """
    if value is None or pd.isna(value):
        return []
    text = str(value).strip()
    if not text or text.lower() in {'nan', 'none', 'null'}:
        return []
    for sep in [';', ',', '|', '/', ':']:
        text = text.replace(sep, '_')
    parts = [p.strip().upper() for p in text.split('_') if p.strip()]
    # Common UniProt protein-name suffixes should not become pseudo genes.
    parts = [p for p in parts if p not in {'HUMAN', 'MOUSE', 'RAT', 'BOVIN', 'PIG', 'YEAST'}]
    # Keep plausible HGNC-like tokens and avoid exploding free-text annotations.
    parts = [p for p in parts if any(ch.isalpha() for ch in p) and len(p) <= 30]
    return sorted(set(parts))


def normalize_ligand_receptor_table(df: pd.DataFrame, database: str = 'unknown') -> pd.DataFrame:
    """Normalize heterogeneous LR resources to ligand/receptor/sources/references.

    Supported inputs include OmniPath-style ``source_genesymbol``/``target_genesymbol``,
    CellChat-style ``ligand``/``receptor``, and CellPhoneDB-style ``partner_a``/
    ``partner_b`` tables after complex/member expansion.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=['ligand', 'receptor', 'sources', 'references', 'database', 'evidence'])
    x = df.copy()
    lig_col = _first_present(x, ['ligand', 'source_genesymbol', 'source', 'from', 'partner_a', 'gene_a', 'interactor_a'])
    rec_col = _first_present(x, ['receptor', 'target_genesymbol', 'target', 'to', 'partner_b', 'gene_b', 'interactor_b'])
    if lig_col is None or rec_col is None:
        raise ValueError(f'Ligand-receptor prior must contain ligand/receptor-like columns: {list(x.columns)}')
    source_col = _first_present(x, ['sources', 'source_database', 'annotation_strategy', 'database'])
    ref_col = _first_present(x, ['references', 'reference', 'pmid', 'pubmed_id'])
    rows = []
    for _, r in x.iterrows():
        ligs = _split_complex_symbol(r[lig_col])
        recs = _split_complex_symbol(r[rec_col])
        if not ligs or not recs:
            continue
        src = str(r[source_col]) if source_col is not None and pd.notna(r[source_col]) else database
        ref = str(r[ref_col]) if ref_col is not None and pd.notna(r[ref_col]) else ''
        for lig in ligs:
            for rec in recs:
                if lig == rec:
                    continue
                rows.append({
                    'ligand': lig,
                    'receptor': rec,
                    'sources': src,
                    'references': ref,
                    'database': database,
                    'evidence': str(r.get('evidence', '')) if isinstance(r, pd.Series) else '',
                })
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=['ligand', 'receptor', 'sources', 'references', 'database', 'evidence'])
    out = out.drop_duplicates(['ligand', 'receptor', 'database']).reset_index(drop=True)
    return out


def load_ligand_receptor_prior(path: str | Path) -> pd.DataFrame:
    """Load a ligand-receptor prior table with flexible column names.

    Accepted input columns include OmniPath-style ``source_genesymbol``/
    ``target_genesymbol`` or generic ``ligand``/``receptor`` / ``source``/``target``.
    Gene symbols are upper-cased to improve cross-platform matching. Complex-like
    entries are expanded to subunit-level pairs.
    """
    path = Path(path)
    sep = ',' if path.suffix.lower() == '.csv' else '\t'
    df = pd.read_csv(path, sep=sep, low_memory=False)
    return normalize_ligand_receptor_table(df, database=path.stem)


def merge_ligand_receptor_priors(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Merge LR prior frames while preserving database provenance."""
    clean = [f.copy() for f in frames if f is not None and not f.empty]
    if not clean:
        return pd.DataFrame(columns=['ligand', 'receptor', 'sources', 'references', 'databases', 'n_databases'])
    df = pd.concat(clean, ignore_index=True)
    for c in ['sources', 'references', 'database', 'evidence']:
        if c not in df.columns:
            df[c] = ''
    out = df.groupby(['ligand', 'receptor'], as_index=False).agg(
        sources=('sources', lambda s: ';'.join(sorted(set(str(x) for x in s if pd.notna(x) and str(x) not in {'', 'nan'})))),
        references=('references', lambda s: ';'.join(sorted(set(str(x) for x in s if pd.notna(x) and str(x) not in {'', 'nan'})))[:5000]),
        databases=('database', lambda s: ';'.join(sorted(set(str(x) for x in s if pd.notna(x) and str(x) not in {'', 'nan'})))),
        n_databases=('database', lambda s: len(set(str(x) for x in s if pd.notna(x) and str(x) not in {'', 'nan'}))),
    )
    out = out.sort_values(['n_databases', 'ligand', 'receptor'], ascending=[False, True, True]).reset_index(drop=True)
    return out
