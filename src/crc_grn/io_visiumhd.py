from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Dict, Optional
import gzip
import yaml
import pandas as pd
import h5py


POSITION_COLS = [
    'barcode',
    'in_tissue',
    'array_row',
    'array_col',
    'pxl_row_in_fullres',
    'pxl_col_in_fullres',
]


def load_sample_map(config_path: str | Path) -> Dict[str, str]:
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    return cfg['sample_to_gsm']


def _candidate_names(sample: str, basename: str, sample_to_gsm: Dict[str, str]) -> list[str]:
    gsm = sample_to_gsm[sample]
    base = basename.replace('.gz', '')
    cands = [
        f'{sample}_{basename}',
        f'{gsm}_{sample}_{basename}',
        f'{sample}_{base}',
        f'{gsm}_{sample}_{base}',
    ]
    if basename.endswith('tissue_positions.parquet.gz'):
        cands.extend([
            f'{sample}_tissue_positions.parquet',
            f'{gsm}_{sample}_tissue_positions.parquet',
            f'{sample}_tissue_positions.csv.gz',
            f'{gsm}_{sample}_tissue_positions.csv.gz',
            f'{sample}_tissue_positions.csv',
            f'{gsm}_{sample}_tissue_positions.csv',
            f'{sample}_tissue_positions_list.csv.gz',
            f'{gsm}_{sample}_tissue_positions_list.csv.gz',
            f'{sample}_tissue_positions_list.csv',
            f'{gsm}_{sample}_tissue_positions_list.csv',
        ])
    if basename.endswith('Metadata.parquet.gz'):
        cands.extend([
            f'{sample}_Metadata.parquet',
            f'{gsm}_{sample}_Metadata.parquet',
            f'{sample}_metadata.parquet.gz',
            f'{gsm}_{sample}_metadata.parquet.gz',
            f'{sample}_metadata.parquet',
            f'{gsm}_{sample}_metadata.parquet',
        ])
    if basename.endswith('scalefactors_json.json.gz'):
        cands.extend([
            f'{sample}_scalefactors_json.json',
            f'{gsm}_{sample}_scalefactors_json.json',
            f'{sample}_scalefactors.json.gz',
            f'{gsm}_{sample}_scalefactors.json.gz',
            f'{sample}_scalefactors.json',
            f'{gsm}_{sample}_scalefactors.json',
        ])
    if basename.endswith('tissue_lowres_image.png.gz'):
        cands.extend([
            f'{sample}_tissue_lowres_image.png',
            f'{gsm}_{sample}_tissue_lowres_image.png',
            f'{sample}_tissue_hires_image.png.gz',
            f'{gsm}_{sample}_tissue_hires_image.png.gz',
            f'{sample}_tissue_hires_image.png',
            f'{gsm}_{sample}_tissue_hires_image.png',
        ])
    return cands


def resolve_file(sample: str, basename: str, root: str | Path, sample_to_gsm: Dict[str, str]) -> Optional[Path]:
    root = Path(root)
    for name in _candidate_names(sample, basename, sample_to_gsm):
        p = root / name
        if p.exists():
            return p
    for pattern in [f'*{sample}*{basename.replace(".gz", "")}', f'*{sample}*{basename}', f'{sample}*']:
        hits = sorted(root.glob(pattern))
        if hits:
            return hits[0]
    return None


def _read_parquet_any(path: Path) -> pd.DataFrame:
    try:
        return pd.read_parquet(path)
    except Exception:
        if path.name.endswith('.parquet.gz'):
            with gzip.open(path, 'rb') as f:
                raw = f.read()
            return pd.read_parquet(BytesIO(raw))
        raise


def _normalize_positions(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rename = {c: c.strip() for c in out.columns if isinstance(c, str)}
    out = out.rename(columns=rename)
    if out.shape[1] >= 6 and not set(POSITION_COLS).issubset(out.columns):
        unnamed = [c for c in out.columns if str(c).startswith('Unnamed:')]
        if unnamed or list(out.columns[:6]) != POSITION_COLS:
            maybe = out.iloc[:, :6].copy()
            maybe.columns = POSITION_COLS
            out = maybe.join(out.iloc[:, 6:])
    if set(POSITION_COLS).issubset(out.columns):
        out['barcode'] = out['barcode'].astype(str)
    else:
        lower = {str(c).lower(): c for c in out.columns}
        if 'barcode' in lower:
            out = out.rename(columns={lower['barcode']: 'barcode'})
    return out


def read_positions_parquet(sample: str, root: str | Path, sample_to_gsm: Dict[str, str]) -> pd.DataFrame:
    p = resolve_file(sample, 'tissue_positions.parquet.gz', root, sample_to_gsm)
    if p is None:
        raise FileNotFoundError(f'No tissue_positions for {sample}')
    name = p.name.lower()
    if name.endswith('.parquet') or name.endswith('.parquet.gz'):
        pos = _read_parquet_any(p)
    else:
        try:
            pos = pd.read_csv(p)
        except Exception:
            pos = pd.read_csv(p, header=None)
        if pos.shape[1] == 1:
            pos = pd.read_csv(p, header=None)
        if pos.shape[1] >= 6 and 'barcode' not in pos.columns:
            pos = pos.iloc[:, :6].copy()
            pos.columns = POSITION_COLS
    return _normalize_positions(pos)


def read_metadata_parquet(sample: str, root: str | Path, sample_to_gsm: Dict[str, str]) -> pd.DataFrame:
    p = resolve_file(sample, 'Metadata.parquet.gz', root, sample_to_gsm)
    if p is None:
        raise FileNotFoundError(f'No Metadata for {sample}')
    return _read_parquet_any(p) if p.name.endswith('.parquet.gz') or p.suffix == '.parquet' else pd.read_csv(p)


def read_visium_h5_summary(sample: str, root: str | Path, sample_to_gsm: Dict[str, str]) -> dict:
    p = resolve_file(sample, 'filtered_feature_bc_matrix.h5', root, sample_to_gsm)
    if p is None:
        raise FileNotFoundError(f'No filtered_feature_bc_matrix.h5 for {sample}')
    out = {'path': str(p), 'n_features': None, 'n_barcodes': None}
    with h5py.File(p, 'r') as h5:
        if 'matrix' in h5 and 'shape' in h5['matrix']:
            shape = h5['matrix']['shape'][()]
            out['n_features'] = int(shape[0])
            out['n_barcodes'] = int(shape[1])
    return out


def summarize_root(root: str | Path, config_path: str | Path) -> pd.DataFrame:
    sample_to_gsm = load_sample_map(config_path)
    rows = []
    for sample, gsm in sample_to_gsm.items():
        for basename in [
            'filtered_feature_bc_matrix.h5',
            'Metadata.parquet.gz',
            'scalefactors_json.json.gz',
            'tissue_lowres_image.png.gz',
            'tissue_positions.parquet.gz',
        ]:
            p = resolve_file(sample, basename, root, sample_to_gsm)
            rows.append({'sample': sample, 'gsm': gsm, 'basename': basename, 'exists': p is not None, 'path': str(p) if p else None})
    return pd.DataFrame(rows)
