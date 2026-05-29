#!/usr/bin/env python
"""Download and normalize real human ligand-receptor priors.

Outputs are written to resources/ccc_prior/:
- omnipath_ligand_receptor_human.tsv
- cellphonedb_ligand_receptor_human.tsv, when parsable
- cellchat_ligand_receptor_human.tsv, when parsable
- ligand_receptor_human.tsv, merged preferred file used by the pipeline

The script is deliberately tolerant: unavailable resources are skipped, but at least
one real database must be parsed unless --allow-seed-fallback is provided.
"""
from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

import pandas as pd
try:
    import requests  # type: ignore
except Exception:  # pragma: no cover
    requests = None

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from crc_grn.prior import normalize_ligand_receptor_table, merge_ligand_receptor_priors, load_ligand_receptor_prior

OMNIPATH_URLS = [
    # genesymbols=yes is important: otherwise some OmniPath deployments return UniProt IDs only.
    'https://omnipathdb.org/interactions?datasets=ligrecextra&organisms=9606&genesymbols=yes&fields=sources,references&format=tsv',
    'https://omnipathdb.org/interactions?datasets=omnipath,ligrecextra&organisms=9606&genesymbols=yes&fields=sources,references&format=tsv',
    # Backward-compatible fallback for older web-service deployments.
    'https://omnipathdb.org/interactions?datasets=ligrecextra&organisms=9606&fields=sources,references&format=tsv',
]

# GitHub repository layouts have changed across CellPhoneDB versions. Try several.
CPDB_BASES = [
    # v5 data repository mirrors. Older archived package-layout URLs are intentionally
    # not tried by default because some corporate/proxy networks keep these raw GitHub
    # requests open for a long time and make the notebook appear frozen.
    'https://raw.githubusercontent.com/ventolab/cellphonedb-data/master/data',
    'https://raw.githubusercontent.com/ventolab/cellphonedb-data/main/data',
    'https://raw.githubusercontent.com/Teichlab/cellphonedb-data/master/data',
    'https://raw.githubusercontent.com/Teichlab/cellphonedb-data/main/data',
]

CELLCHAT_RDA_URLS = [
    'https://github.com/jinworks/CellChat/raw/main/data/CellChatDB.human.rda',
    'https://github.com/sqjin/CellChat/raw/master/data/CellChatDB.human.rda',
]


def download_bytes(url: str, timeout: int = 20) -> bytes:
    headers = {'User-Agent': 'crc-grn-prior-downloader/1.1'}
    if requests is not None:
        r = requests.get(url, headers=headers, timeout=(min(timeout, 10), timeout))
        r.raise_for_status()
        return r.content
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def try_read_table_from_url(url: str, sep: str = '\t', timeout: int = 20) -> pd.DataFrame | None:
    try:
        data = download_bytes(url, timeout=timeout)
        if not data or data[:80].lower().startswith(b'<!doctype'):
            return None
        return pd.read_csv(io.BytesIO(data), sep=sep, low_memory=False)
    except Exception:
        return None


def download_omnipath(outdir: Path, timeout: int = 20) -> pd.DataFrame:
    for url in OMNIPATH_URLS:
        df = try_read_table_from_url(url, sep='\t', timeout=timeout)
        if df is None or df.empty:
            continue
        # OmniPath columns are usually source_genesymbol/target_genesymbol.
        out = normalize_ligand_receptor_table(df, database='OmniPath')
        if not out.empty:
            path = outdir / 'omnipath_ligand_receptor_human.tsv'
            out.to_csv(path, sep='\t', index=False)
            return out
    return pd.DataFrame()


def _first_col(df: pd.DataFrame, names: list[str]) -> str | None:
    lookup = {str(c).lower(): c for c in df.columns}
    for n in names:
        if n in df.columns:
            return n
        if n.lower() in lookup:
            return lookup[n.lower()]
    return None


def _split_symbols(x) -> list[str]:
    if x is None or pd.isna(x):
        return []
    text = str(x).strip()
    for sep in [';', ',', '|', '/', ':']:
        text = text.replace(sep, '_')
    parts = [p.strip().upper() for p in text.split('_') if p.strip()]
    parts = [p for p in parts if p not in {'HUMAN', 'MOUSE', 'RAT', 'BOVIN', 'PIG', 'YEAST'}]
    return sorted(set(p for p in parts if any(ch.isalpha() for ch in p) and len(p) <= 30))


def _build_cpdb_maps(protein: pd.DataFrame | None, gene: pd.DataFrame | None, complex_df: pd.DataFrame | None) -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = {}
    if gene is not None and not gene.empty:
        gene_name = _first_col(gene, ['gene_name', 'hgnc_symbol', 'name', 'symbol'])
        protein_id = _first_col(gene, ['protein_id', 'id_protein', 'uniprot', 'ensembl'])
        if gene_name is not None:
            for _, r in gene.iterrows():
                syms = _split_symbols(r[gene_name])
                for val in [r.get(gene_name), r.get(protein_id) if protein_id else None]:
                    if val is not None and not pd.isna(val):
                        mapping[str(val)] = syms
    if protein is not None and not protein.empty:
        pid = _first_col(protein, ['id_protein', 'protein_id', 'uniprot', 'name'])
        pname = _first_col(protein, ['protein_name', 'gene_name', 'hgnc_symbol', 'symbol', 'name'])
        if pid is not None and pname is not None:
            for _, r in protein.iterrows():
                pid_key = str(r[pid])
                syms = mapping.get(pid_key, _split_symbols(r[pname]))
                # Do not overwrite UniProt -> HGNC mappings from gene_input.csv with
                # UniProt entry names such as CADH1_HUMAN.
                mapping.setdefault(pid_key, syms)
                mapping[str(r[pname])] = syms
    if complex_df is not None and not complex_df.empty:
        cname = _first_col(complex_df, ['complex_name', 'name', 'id_complex', 'complex_id'])
        comp_cols = [c for c in complex_df.columns if 'uniprot' in str(c).lower() or 'protein' in str(c).lower() or 'subunit' in str(c).lower()]
        if cname is not None and comp_cols:
            for _, r in complex_df.iterrows():
                syms = []
                for c in comp_cols:
                    key = str(r[c]) if pd.notna(r[c]) else ''
                    syms.extend(mapping.get(key, _split_symbols(key)))
                syms = sorted(set(syms))
                mapping[str(r[cname])] = syms
    return mapping


def _read_first_existing(base: str, stems: list[str], timeout: int = 20) -> pd.DataFrame | None:
    for stem in stems:
        for sep in [',', '\t']:
            df = try_read_table_from_url(f'{base}/{stem}', sep=sep, timeout=timeout)
            if df is not None and not df.empty:
                return df
    return None


def download_cellphonedb(outdir: Path, timeout: int = 20) -> pd.DataFrame:
    for base in CPDB_BASES:
        interaction = _read_first_existing(base, ['interaction_input.csv', 'interaction.csv'], timeout=timeout)
        if interaction is None or interaction.empty:
            continue
        protein = _read_first_existing(base, ['protein_input.csv', 'protein.csv'], timeout=timeout)
        gene = _read_first_existing(base, ['gene_input.csv', 'gene.csv'], timeout=timeout)
        complex_df = _read_first_existing(base, ['complex_input.csv', 'complex.csv'], timeout=timeout)
        pa = _first_col(interaction, ['partner_a', 'protein_name_a', 'gene_a', 'ligand'])
        pb = _first_col(interaction, ['partner_b', 'protein_name_b', 'gene_b', 'receptor'])
        if pa is None or pb is None:
            continue
        mapping = _build_cpdb_maps(protein, gene, complex_df)
        rows = []
        for _, r in interaction.iterrows():
            a_raw, b_raw = str(r[pa]), str(r[pb])
            a = mapping.get(a_raw, _split_symbols(a_raw))
            b = mapping.get(b_raw, _split_symbols(b_raw))
            for lig in a:
                for rec in b:
                    if lig != rec:
                        rows.append({'ligand': lig, 'receptor': rec, 'sources': 'CellPhoneDB', 'references': '', 'database': 'CellPhoneDB', 'evidence': ''})
        out = pd.DataFrame(rows).drop_duplicates() if rows else pd.DataFrame()
        if not out.empty:
            path = outdir / 'cellphonedb_ligand_receptor_human.tsv'
            out.to_csv(path, sep='\t', index=False)
            return out
    return pd.DataFrame()


def download_cellchat(outdir: Path, timeout: int = 20) -> pd.DataFrame:
    rda_path = outdir / 'CellChatDB.human.rda'
    for url in CELLCHAT_RDA_URLS:
        try:
            rda_path.write_bytes(download_bytes(url, timeout=timeout))
            if rda_path.stat().st_size > 1000:
                break
        except Exception:
            continue
    if not rda_path.exists() or rda_path.stat().st_size <= 1000:
        return pd.DataFrame()

    # Prefer R because CellChatDB is an R list stored in .rda.
    r_script = outdir / '_extract_cellchatdb.R'
    csv_path = outdir / '_cellchat_interactions.csv'
    r_script.write_text(f"""
load('{rda_path.as_posix()}')
obj <- if (exists('CellChatDB.human')) CellChatDB.human else get(ls()[1])
interaction <- obj$interaction
write.csv(interaction, '{csv_path.as_posix()}', row.names=FALSE)
""", encoding='utf-8')
    try:
        subprocess.check_call(['Rscript', str(r_script)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        # Optional pyreadr fallback only works when the R list is flattened by pyreadr.
        try:
            import pyreadr  # type: ignore
            res = pyreadr.read_r(str(rda_path))
            for _, obj in res.items():
                if isinstance(obj, pd.DataFrame) and {'ligand', 'receptor'}.issubset(set(map(str.lower, obj.columns))):
                    csv_path.write_text(obj.to_csv(index=False), encoding='utf-8')
                    break
        except Exception:
            pass
    if not csv_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(csv_path, low_memory=False)
    out = normalize_ligand_receptor_table(df, database='CellChatDB')
    if not out.empty:
        path = outdir / 'cellchat_ligand_receptor_human.tsv'
        out.to_csv(path, sep='\t', index=False)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', type=Path, default=ROOT / 'resources' / 'ccc_prior')
    parser.add_argument('--allow-seed-fallback', action='store_true',
                        help='Use ligand_receptor_seed_crc.tsv only when all real database downloads fail. Use for smoke tests, not final analysis.')
    parser.add_argument('--timeout', type=int, default=20, help='Network timeout in seconds for each URL.')
    parser.add_argument('--min-real-pairs', type=int, default=100,
                        help='Minimum number of merged pairs expected when at least one real database succeeds.')
    parser.add_argument('--force', action='store_true',
                        help='Force redownload even if ligand_receptor_human.tsv already looks valid.')
    args = parser.parse_args()
    outdir: Path = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    existing = outdir / 'ligand_receptor_human.tsv'
    if existing.exists() and not args.force:
        try:
            current = pd.read_csv(existing, sep='\t', low_memory=False)
            has_cols = {'ligand', 'receptor'}.issubset(current.columns)
            is_seed_only = False
            if 'databases' in current.columns:
                db_text = ';'.join(current['databases'].dropna().astype(str).unique()).lower()
                is_seed_only = ('seed' in db_text or 'curated_crc_ccc' in db_text) and current.shape[0] < args.min_real_pairs
            if has_cols and current.shape[0] >= args.min_real_pairs and not is_seed_only:
                log = pd.DataFrame([{'database': 'existing_ligand_receptor_human.tsv', 'n_pairs': int(current.shape[0]), 'status': 'reused_existing_valid_prior'}])
                log.to_csv(outdir / 'ccc_prior_download_log.csv', index=False)
                print(log)
                print(f'[ok] existing ligand-receptor prior reused: {existing} shape={current.shape}')
                return
        except Exception:
            pass

    frames = []
    logs = []
    for name, func in [('OmniPath', download_omnipath), ('CellPhoneDB', download_cellphonedb), ('CellChatDB', download_cellchat)]:
        try:
            df = func(outdir, timeout=args.timeout)
            logs.append({'database': name, 'n_pairs': int(df.shape[0]), 'status': 'ok' if not df.empty else 'empty_or_unavailable'})
            if not df.empty:
                df = df.copy()
                df['database'] = name
                frames.append(df)
        except Exception as e:
            logs.append({'database': name, 'n_pairs': 0, 'status': f'failed: {e}'})

    if not frames and args.allow_seed_fallback:
        seed = outdir / 'ligand_receptor_seed_crc.tsv'
        if seed.exists():
            frames.append(load_ligand_receptor_prior(seed))
            logs.append({'database': 'CRC_seed_fallback', 'n_pairs': int(frames[-1].shape[0]), 'status': 'fallback'})

    if not frames:
        pd.DataFrame(logs).to_csv(outdir / 'ccc_prior_download_log.csv', index=False)
        raise SystemExit('No real LR database could be parsed. Check internet access/proxy/SSL, GitHub/OmniPath access, or install R/Rscript for CellChat extraction. For a smoke test only, rerun with --allow-seed-fallback.')

    merged = merge_ligand_receptor_priors(frames)
    real_rows = int(merged.shape[0])
    used_only_seed = all(str(x.get('database', '')).startswith('CRC_seed') for x in logs if x.get('n_pairs', 0))
    if real_rows < args.min_real_pairs and not used_only_seed:
        logs.append({'database': 'quality_check', 'n_pairs': real_rows, 'status': f'warning: merged pairs < min_real_pairs={args.min_real_pairs}'})
    merged.to_csv(outdir / 'ligand_receptor_human.tsv', sep='\t', index=False)
    pd.DataFrame(logs).to_csv(outdir / 'ccc_prior_download_log.csv', index=False)
    print(pd.DataFrame(logs))
    print(f'[ok] merged ligand-receptor prior: {outdir / "ligand_receptor_human.tsv"} shape={merged.shape}')


if __name__ == '__main__':
    main()
