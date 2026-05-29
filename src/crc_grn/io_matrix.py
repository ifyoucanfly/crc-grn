from __future__ import annotations

from pathlib import Path
import gzip
import io
import json
import tarfile
import tempfile
import zipfile
from typing import Iterable, Optional

import h5py
import numpy as np
import pandas as pd
from scipy.io import mmread
from scipy.sparse import csc_matrix


def _open_text_auto(path: str | Path):
    path = Path(path)
    if path.suffix == '.gz':
        return gzip.open(path, 'rt', encoding='utf-8', errors='replace')
    return open(path, 'rt', encoding='utf-8', errors='replace')


def _read_table_auto(path: str | Path, sep: Optional[str] = None, header: Optional[int] = None) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == '.gz':
        with gzip.open(path, 'rt', encoding='utf-8', errors='replace') as f:
            return pd.read_csv(f, sep=sep, header=header)
    return pd.read_csv(path, sep=sep, header=header)


def _find_first(root: str | Path, patterns: Iterable[str]) -> Optional[Path]:
    root = Path(root)
    for pattern in patterns:
        hits = sorted(root.rglob(pattern))
        if hits:
            return hits[0]
    return None


def _parse_features_table(path: str | Path) -> list[str]:
    df = _read_table_auto(path, sep='\t', header=None)
    if df.shape[1] >= 2:
        genes = df.iloc[:, 1].astype(str).tolist()
    else:
        genes = df.iloc[:, 0].astype(str).tolist()
    return genes


def _parse_barcodes_table(path: str | Path) -> list[str]:
    df = _read_table_auto(path, sep='\t', header=None)
    return df.iloc[:, 0].astype(str).tolist()




def _maybe_decompress_bytes(data: bytes) -> bytes:
    if data[:2] == b"\x1f\x8b":
        try:
            return gzip.decompress(data)
        except Exception:
            return data
    return data


def _read_table_member_bytes(data: bytes, sep: Optional[str] = None, header: Optional[int] = None) -> pd.DataFrame:
    payload = _maybe_decompress_bytes(data)
    return pd.read_csv(io.BytesIO(payload), sep=sep, header=header)


def _read_csv_member_bytes(data: bytes, **kwargs) -> pd.DataFrame:
    payload = _maybe_decompress_bytes(data)
    return pd.read_csv(io.BytesIO(payload), **kwargs)


def _looks_like_numeric_label(x) -> bool:
    try:
        float(str(x))
        return True
    except Exception:
        return False


def _standard_positions_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if df.shape[1] >= 6:
        df = df.iloc[:, :6].copy()
        df.columns = ['barcode', 'in_tissue', 'array_row', 'array_col', 'pxl_row_in_fullres', 'pxl_col_in_fullres']
    return df

def _read_mtx_auto(path: str | Path):
    path = Path(path)
    if path.suffix == '.gz':
        with gzip.open(path, 'rb') as f:
            return mmread(f).tocsc()
    return mmread(path).tocsc()




def _read_mtx_from_bytes(data: bytes):
    bio = io.BytesIO(_maybe_decompress_bytes(data))
    return mmread(bio).tocsc()


def _read_table_from_bytes(data: bytes, sep: Optional[str] = None, header: Optional[int] = None) -> pd.DataFrame:
    return _read_table_member_bytes(data, sep=sep, header=header)


def _archive_sample_name(path: str | Path) -> str:
    path = Path(path)
    name = path.name
    for suf in ['.tar.gz', '.tgz', '.tar', '.zip', '.gz']:
        if name.endswith(suf):
            return name[:-len(suf)]
    return path.stem

def _normalize_positions_df(pos: pd.DataFrame) -> pd.DataFrame:
    pos = pos.copy()
    if pos.shape[1] >= 6:
        first_cols = list(pos.columns[:6])
        # headerless 10x tissue_positions_list accidentally parsed with the first row as header
        if all(_looks_like_numeric_label(c) for c in first_cols[1:6]):
            first_row = pd.DataFrame([first_cols[:6]], columns=first_cols[:6])
            pos = pd.concat([first_row, pos.iloc[:, :6]], axis=0, ignore_index=True)
            pos = _standard_positions_columns(pos)

    cols = {str(c).lower(): c for c in pos.columns}
    if 'barcode' in cols:
        pos = pos.rename(columns={cols['barcode']: 'barcode'})
    elif pos.shape[1] >= 1 and pos.columns[0] not in {'barcode', 'x', 'y'}:
        pos = pos.rename(columns={pos.columns[0]: 'barcode'})

    canonical = {'barcode', 'in_tissue', 'array_row', 'array_col', 'pxl_row_in_fullres', 'pxl_col_in_fullres'}
    if canonical.issubset(set(pos.columns)):
        pos = pos.rename(columns={'pxl_col_in_fullres': 'x', 'pxl_row_in_fullres': 'y'})
    else:
        for xcand in ['pxl_col_in_fullres', 'pixel_x', 'x_centroid', 'x', 'pxl_col', 'imagecol']:
            if xcand in pos.columns:
                pos = pos.rename(columns={xcand: 'x'})
                break
        for ycand in ['pxl_row_in_fullres', 'pixel_y', 'y_centroid', 'y', 'pxl_row', 'imagerow']:
            if ycand in pos.columns:
                pos = pos.rename(columns={ycand: 'y'})
                break
    if 'barcode' not in pos.columns:
        pos = pos.reset_index().rename(columns={'index': 'barcode'})
    if 'x' not in pos.columns or 'y' not in pos.columns:
        raise ValueError(f'Could not infer x/y columns from {list(pos.columns)}')
    pos['barcode'] = pos['barcode'].astype(str)
    pos = pos[pos['barcode'].str.lower() != 'barcode'].copy()
    pos['x'] = pd.to_numeric(pos['x'], errors='coerce')
    pos['y'] = pd.to_numeric(pos['y'], errors='coerce')
    pos = pos.dropna(subset=['barcode', 'x', 'y'])
    return pos[['barcode', 'x', 'y']].drop_duplicates('barcode')

def find_positions_file(root: str | Path) -> Optional[Path]:
    return _find_first(root, [
        '*tissue_positions.parquet',
        '*tissue_positions.parquet.gz',
        '*tissue_positions.csv',
        '*tissue_positions.csv.gz',
        '*tissue_positions_list.csv',
        '*tissue_positions_list.csv.gz',
    ])


def read_positions_any(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == '.parquet' or path.name.endswith('.parquet.gz'):
        pos = pd.read_parquet(path)
        return _normalize_positions_df(pos)
    for sep in [',', '	']:
        try:
            pos = pd.read_csv(path, sep=sep)
            norm = _normalize_positions_df(pos)
            if not norm.empty:
                return norm
        except Exception:
            pass
    for sep in [',', '	']:
        try:
            pos = _read_table_auto(path, sep=sep, header=None)
            if pos.shape[1] >= 6:
                pos = _standard_positions_columns(pos)
            norm = _normalize_positions_df(pos)
            if not norm.empty:
                return norm
        except Exception:
            pass
    raise ValueError(f'Unable to parse positions file: {path}')

def _subset_expr_from_matrix(X, gene_names: list[str], barcodes: list[str], genes: Optional[Iterable[str]] = None, prefix: str = ''):
    gene_to_ix = {str(g): i for i, g in enumerate(gene_names)}
    gene_to_ix_upper = {str(g).upper(): i for i, g in enumerate(gene_names)}
    if genes is None:
        use_ix = list(range(len(gene_names)))
        use_genes = list(gene_names)
    else:
        use_ix = []
        use_genes = []
        seen = set()
        for g in genes:
            gs = str(g)
            if gs in gene_to_ix:
                ix = gene_to_ix[gs]
            elif gs.upper() in gene_to_ix_upper:
                ix = gene_to_ix_upper[gs.upper()]
            else:
                continue
            if ix in seen:
                continue
            seen.add(ix)
            use_ix.append(ix)
            use_genes.append(gene_names[ix])
    if not use_genes:
        raise ValueError('None of the requested genes found in matrix source')
    feat_ix = np.array(use_ix)
    sub = X[feat_ix, :].T.toarray().astype(float)
    expr = pd.DataFrame(sub, index=[f'{prefix}{b}' for b in barcodes], columns=use_genes)
    return expr


def _read_mex_archive(archive_path: str | Path, genes: Optional[Iterable[str]] = None, prefix: str = '', require_positions: bool = True):
    archive_path = Path(archive_path)
    if archive_path.suffix == '.zip':
        with zipfile.ZipFile(archive_path) as zf:
            members = [m for m in zf.namelist() if not m.endswith('/')]
            barcodes_member = next((m for m in members if m.endswith('barcodes.tsv.gz') or m.endswith('barcodes.tsv')), None)
            features_member = next((m for m in members if m.endswith('features.tsv.gz') or m.endswith('features.tsv')), None)
            matrix_member = next((m for m in members if m.endswith('matrix.mtx.gz') or m.endswith('matrix.mtx')), None)
            pos_member = next((m for m in members if any(m.endswith(s) for s in ['tissue_positions.parquet', 'tissue_positions.parquet.gz', 'tissue_positions.csv', 'tissue_positions.csv.gz', 'tissue_positions_list.csv', 'tissue_positions_list.csv.gz'])), None)
            if barcodes_member is None or features_member is None or matrix_member is None:
                raise FileNotFoundError(f'Missing matrix/barcodes/features inside {archive_path.name}')
            with zf.open(barcodes_member) as f:
                barcodes = _parse_barcodes_table(io.BytesIO(f.read())) if False else None
            # parse tables explicitly from bytes for zip members
            with zf.open(barcodes_member) as f:
                barcodes_df = _read_table_member_bytes(f.read(), sep='	', header=None)
            barcodes = barcodes_df.iloc[:, 0].astype(str).tolist()
            with zf.open(features_member) as f:
                features_df = _read_table_member_bytes(f.read(), sep='	', header=None)
            gene_names = features_df.iloc[:, 1].astype(str).tolist() if features_df.shape[1] >= 2 else features_df.iloc[:, 0].astype(str).tolist()
            with zf.open(matrix_member) as f:
                X = _read_mtx_from_bytes(f.read())
            if X.shape[0] != len(gene_names):
                X = X.T.tocsc()
            expr = _subset_expr_from_matrix(X, gene_names, barcodes, genes=genes, prefix=prefix)
            coords = None
            if require_positions and pos_member is not None:
                if pos_member.endswith('.parquet') or pos_member.endswith('.parquet.gz'):
                    tmp = _extract_zip_member_to_temp(zf, pos_member)
                    try:
                        pos = pd.read_parquet(tmp)
                    finally:
                        tmp.unlink(missing_ok=True)
                else:
                    with zf.open(pos_member) as f:
                        pos = _read_csv_member_bytes(f.read())
                        if pos.shape[1] == 1:
                            with zf.open(pos_member) as f2:
                                if pos_member.endswith('.gz'):
                                    pos = pd.read_csv(gzip.GzipFile(fileobj=f2), sep=',', header=None)
                                else:
                                    pos = pd.read_csv(f2, sep=',', header=None)
                            if pos.shape[1] == 1:
                                with zf.open(pos_member) as f3:
                                    pos = _read_csv_member_bytes(f3.read(), sep='	', header=None)
                pos = _normalize_positions_df(pos)
                pos['barcode'] = [f'{prefix}{b}' for b in pos['barcode'].astype(str)]
                coords = pos.set_index('barcode')[['x','y']]
    else:
        mode = 'r:gz' if archive_path.name.endswith(('.tar.gz', '.tgz')) else 'r:'
        with tarfile.open(archive_path, mode) as tf:
            members = [m for m in tf.getmembers() if m.isfile()]
            names = [m.name for m in members]
            def _pick(cands):
                return next((m for m in members if any(m.name.endswith(c) for c in cands)), None)
            barcodes_member = _pick(['barcodes.tsv.gz', 'barcodes.tsv'])
            features_member = _pick(['features.tsv.gz', 'features.tsv'])
            matrix_member = _pick(['matrix.mtx.gz', 'matrix.mtx'])
            pos_member = _pick(['tissue_positions.parquet', 'tissue_positions.parquet.gz', 'tissue_positions.csv', 'tissue_positions.csv.gz', 'tissue_positions_list.csv', 'tissue_positions_list.csv.gz'])
            if barcodes_member is None or features_member is None or matrix_member is None:
                raise FileNotFoundError(f'Missing matrix/barcodes/features inside {archive_path.name}')
            with tf.extractfile(barcodes_member) as f:
                data = f.read()
                barcodes_df = _read_table_member_bytes(data, sep='	', header=None)
            barcodes = barcodes_df.iloc[:, 0].astype(str).tolist()
            with tf.extractfile(features_member) as f:
                data = f.read()
                features_df = _read_table_member_bytes(data, sep='	', header=None)
            gene_names = features_df.iloc[:, 1].astype(str).tolist() if features_df.shape[1] >= 2 else features_df.iloc[:, 0].astype(str).tolist()
            with tf.extractfile(matrix_member) as f:
                data = f.read()
                X = _read_mtx_from_bytes(data)
            if X.shape[0] != len(gene_names):
                X = X.T.tocsc()
            expr = _subset_expr_from_matrix(X, gene_names, barcodes, genes=genes, prefix=prefix)
            coords = None
            if require_positions and pos_member is not None:
                with tf.extractfile(pos_member) as f:
                    data = f.read()
                if pos_member.name.endswith('.parquet') or pos_member.name.endswith('.parquet.gz'):
                    suffix = '.parquet.gz' if pos_member.name.endswith('.parquet.gz') else '.parquet'
                    fd,tmp = tempfile.mkstemp(suffix=suffix)
                    Path(tmp).unlink(missing_ok=True)
                    tmp_path = Path(tmp)
                    try:
                        tmp_path.write_bytes(data)
                        pos = pd.read_parquet(tmp_path)
                    finally:
                        tmp_path.unlink(missing_ok=True)
                else:
                    pos = _read_csv_member_bytes(data)
                    if pos.shape[1] == 1:
                        pos = _read_csv_member_bytes(data, sep=',', header=None)
                    if pos.shape[1] == 1:
                        pos = _read_csv_member_bytes(data, sep='	', header=None)
                pos = _normalize_positions_df(pos)
                pos['barcode'] = [f'{prefix}{b}' for b in pos['barcode'].astype(str)]
                coords = pos.set_index('barcode')[['x','y']]
    if require_positions and coords is None:
        raise FileNotFoundError(f'Missing positions inside {archive_path.name}')
    if not require_positions:
        return expr, None
    if coords is not None:
        common = expr.index.intersection(coords.index)
        expr = expr.loc[common]
        coords = coords.loc[common, ['x', 'y']]
    return expr, coords


def read_mex_folder(folder: str | Path, genes: Optional[Iterable[str]] = None, prefix: str = '', require_positions: bool = True):
    folder = Path(folder)
    if folder.is_file() and (folder.suffix == '.zip' or folder.name.endswith(('.tar.gz', '.tgz', '.tar'))):
        return _read_mex_archive(folder, genes=genes, prefix=prefix, require_positions=require_positions)
    barcodes_path = _find_first(folder, ['*barcodes.tsv.gz', '*barcodes.tsv'])
    features_path = _find_first(folder, ['*features.tsv.gz', '*features.tsv'])
    matrix_path = _find_first(folder, ['*matrix.mtx.gz', '*matrix.mtx'])
    pos_path = find_positions_file(folder)
    if barcodes_path is None or features_path is None or matrix_path is None:
        raise FileNotFoundError(f'Missing matrix/barcodes/features under {folder}')
    if require_positions and pos_path is None:
        raise FileNotFoundError(f'Missing positions under {folder}')
    barcodes = _parse_barcodes_table(barcodes_path)
    feature_names = _parse_features_table(features_path)
    X = _read_mtx_auto(matrix_path)
    if X.shape[0] == len(feature_names):
        gene_names = feature_names
    else:
        X = X.T.tocsc()
        gene_names = feature_names
    expr = _subset_expr_from_matrix(X, gene_names, barcodes, genes=genes, prefix=prefix)
    if not require_positions or pos_path is None:
        return expr, None
    pos = read_positions_any(pos_path)
    pos['barcode'] = [f'{prefix}{b}' for b in pos['barcode'].astype(str)]
    pos = pos.set_index('barcode')
    common = expr.index.intersection(pos.index)
    expr = expr.loc[common]
    coords = pos.loc[common, ['x', 'y']]
    return expr, coords


def list_mex_sources(root: str | Path) -> list[Path]:
    root = Path(root)
    if not root.exists():
        return []
    sources = []
    # extracted directories
    for p in sorted(root.iterdir()):
        if p.is_dir() and _find_first(p, ['*barcodes.tsv.gz', '*barcodes.tsv']) is not None and _find_first(p, ['*features.tsv.gz', '*features.tsv']) is not None and _find_first(p, ['*matrix.mtx.gz', '*matrix.mtx']) is not None:
            sources.append(p)
    # direct archives like Ajou_Visium_P1.tar.gz
    for p in sorted(root.iterdir()):
        if p.is_file() and (p.name.endswith('.tar.gz') or p.name.endswith('.tgz') or p.name.endswith('.zip') or p.suffix == '.tar'):
            sources.append(p)
    return sources


def _zip_has_members(path: Path, suffixes: tuple[str, ...]) -> bool:
    try:
        with zipfile.ZipFile(path) as zf:
            members = zf.namelist()
        return any(any(m.endswith(s) for s in suffixes) for m in members)
    except Exception:
        return False


def list_xenium_runs(root: str | Path) -> list[Path]:
    """Find complete Xenium runs under a root directory.

    A run is considered high-resolution parseable only if both a cell matrix and a
    cell centroid table are available. The search is recursive and supports extracted
    ``outs`` folders as well as direct ``*_outs.zip`` archives.
    """
    root = Path(root)
    if not root.exists():
        return []
    runs: list[Path] = []
    seen: set[str] = set()
    # Prefer direct children for stable sample names, then recursive candidates.
    candidates = list(root.iterdir()) + [p for p in root.rglob('*') if p != root]
    for p in sorted(candidates):
        if str(p) in seen or p.name.endswith('.zarr'):
            continue
        if p.is_dir():
            has_h5 = _find_first(p, ['*cell_feature_matrix.h5']) is not None
            has_cells = _find_first(p, ['*cells.csv.gz', '*cells.csv']) is not None
            if has_h5 and has_cells:
                runs.append(p)
                seen.add(str(p))
        elif p.is_file() and p.suffix == '.zip':
            has_h5 = _zip_has_members(p, ('cell_feature_matrix.h5',))
            has_cells = _zip_has_members(p, ('cells.csv.gz', 'cells.csv'))
            if has_h5 and has_cells:
                runs.append(p)
                seen.add(str(p))
    # remove nested runs contained inside already selected parent directories
    unique = []
    for p in runs:
        if any(p != q and str(p).startswith(str(q) + '/') for q in runs if q.is_dir()):
            continue
        unique.append(p)
    return unique


def _read_10x_h5_from_path(path: str | Path, selected_genes: Optional[Iterable[str]] = None):
    with h5py.File(path, 'r') as h5:
        grp = h5['matrix']
        data = grp['data'][()]
        indices = grp['indices'][()]
        indptr = grp['indptr'][()]
        shape = tuple(grp['shape'][()])
        X = csc_matrix((data, indices, indptr), shape=shape)
        barcodes = [b.decode() if isinstance(b, (bytes, bytearray)) else str(b) for b in grp['barcodes'][()]]
        f = grp['features']
        gene_names = None
        for key in ['name', 'gene_names', 'id']:
            if key in f:
                gene_names = [b.decode() if isinstance(b, (bytes, bytearray)) else str(b) for b in f[key][()]]
                break
        if gene_names is None:
            raise ValueError('No feature names found in H5')
    gene_to_ix = {g: i for i, g in enumerate(gene_names)}
    gene_to_ix_upper = {str(g).upper(): i for i, g in enumerate(gene_names)}
    if selected_genes is None:
        use_ix = list(range(len(gene_names)))
        use_genes = list(gene_names)
    else:
        use_ix = []
        use_genes = []
        seen = set()
        for g in selected_genes:
            gs = str(g)
            if gs in gene_to_ix:
                ix = gene_to_ix[gs]
            elif gs.upper() in gene_to_ix_upper:
                ix = gene_to_ix_upper[gs.upper()]
            else:
                continue
            if ix in seen:
                continue
            seen.add(ix)
            use_ix.append(ix)
            use_genes.append(gene_names[ix])
    if not use_genes:
        raise ValueError('No selected genes found in H5')
    feat_ix = np.array(use_ix)
    sub = X[feat_ix, :].T.toarray().astype(float)
    expr = pd.DataFrame(sub, index=barcodes, columns=use_genes)
    return expr


def _extract_zip_member_to_temp(zf: zipfile.ZipFile, member: str) -> Path:
    suffix = Path(member).suffix
    if member.endswith('.h5'):
        suffix = '.h5'
    fd, tmp = tempfile.mkstemp(suffix=suffix)
    Path(tmp).unlink(missing_ok=True)
    out = Path(tmp)
    with zf.open(member) as src, open(out, 'wb') as dst:
        dst.write(src.read())
    return out


def read_xenium_run(run_path: str | Path, genes: Optional[Iterable[str]] = None, prefix: str = ''):
    run_path = Path(run_path)
    if run_path.is_dir():
        h5_path = _find_first(run_path, ['*cell_feature_matrix.h5'])
        cells_path = _find_first(run_path, ['*cells.csv.gz', '*cells.csv'])
        if h5_path is None or cells_path is None:
            raise FileNotFoundError(f'No cell_feature_matrix.h5 or cells.csv under {run_path}')
        expr = _read_10x_h5_from_path(h5_path, genes)
        cells = pd.read_csv(cells_path)
    else:
        with zipfile.ZipFile(run_path) as zf:
            members = zf.namelist()
            h5_member = next((m for m in members if m.endswith('cell_feature_matrix.h5')), None)
            cells_member = next((m for m in members if m.endswith('cells.csv.gz') or m.endswith('cells.csv')), None)
            if h5_member is None or cells_member is None:
                raise FileNotFoundError(f'No Xenium matrix/cells members inside {run_path.name}')
            tmp_h5 = _extract_zip_member_to_temp(zf, h5_member)
            try:
                expr = _read_10x_h5_from_path(tmp_h5, genes)
            finally:
                tmp_h5.unlink(missing_ok=True)
            with zf.open(cells_member) as f:
                if cells_member.endswith('.gz'):
                    cells = pd.read_csv(gzip.GzipFile(fileobj=f))
                else:
                    cells = pd.read_csv(f)
    xcol = next((c for c in ['x_centroid', 'x_center', 'x'] if c in cells.columns), None)
    ycol = next((c for c in ['y_centroid', 'y_center', 'y'] if c in cells.columns), None)
    bcol = next((c for c in ['cell_id', 'barcode'] if c in cells.columns), None)
    if xcol is None or ycol is None or bcol is None:
        raise ValueError(f'Could not infer Xenium centroid columns from {list(cells.columns)}')
    coords = cells[[bcol, xcol, ycol]].copy()
    coords.columns = ['barcode', 'x', 'y']
    coords['barcode'] = coords['barcode'].astype(str)
    expr.index = expr.index.astype(str)
    expr.index = [f'{prefix}{b}' for b in expr.index]
    coords['barcode'] = [f'{prefix}{b}' for b in coords['barcode']]
    coords = coords.set_index('barcode')
    common = expr.index.intersection(coords.index)
    return expr.loc[common], coords.loc[common, ['x', 'y']]


def read_xenium_gene_panel_json(path: str | Path) -> list[str]:
    path = Path(path)
    if path.suffix == '.gz':
        with gzip.open(path, 'rt', encoding='utf-8', errors='replace') as f:
            data = json.load(f)
    else:
        data = json.loads(path.read_text(encoding='utf-8'))
    genes: set[str] = set()

    def walk(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                lk = k.lower()
                if lk in {'gene', 'gene_name', 'name', 'symbol'} and isinstance(v, str) and 1 <= len(v) <= 30:
                    if any(ch.isalpha() for ch in v):
                        genes.add(v.upper())
                walk(v)
        elif isinstance(obj, list):
            for item in obj:
                walk(item)

    walk(data)
    return sorted(genes)
