from __future__ import annotations

from pathlib import Path
import gzip
from typing import Sequence


def _open_text(path: str | Path):
    path = Path(path)
    if path.suffix == '.gz':
        return gzip.open(path, 'rt', encoding='utf-8', errors='replace')
    return open(path, 'rt', encoding='utf-8', errors='replace')


def extract_gene_names_from_matrix(matrix_path: str | Path) -> list[str]:
    genes: list[str] = []
    seen: set[str] = set()
    with _open_text(matrix_path) as f:
        _ = f.readline()
        for line in f:
            if not line.strip():
                continue
            fields = line.rstrip('\n').split('\t')
            if not fields:
                continue
            g = fields[0].strip().strip('"')
            if not g or g.lower() in {'gene', 'genes', 'gene_name', 'gene_id', 'symbol'}:
                continue
            if g not in seen:
                genes.append(g)
                seen.add(g)
    return genes


def load_sc_reference_gene_sets(matrix_paths: Sequence[str | Path]) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for p in matrix_paths:
        path = Path(p)
        if path.exists():
            out[path.name] = set(extract_gene_names_from_matrix(path))
    return out


def summarize_sc_reference_gene_sets(matrix_paths: Sequence[str | Path]) -> dict:
    gene_sets = load_sc_reference_gene_sets(matrix_paths)
    if not gene_sets:
        return {
            'available': False,
            'files_found': [],
            'n_files': 0,
            'per_file_gene_counts': {},
            'union_n_genes': 0,
            'intersection_n_genes': 0,
            'union_genes': set(),
            'intersection_genes': set(),
        }
    values = list(gene_sets.values())
    union = set().union(*values)
    inter = set(values[0]).intersection(*values[1:]) if len(values) > 1 else set(values[0])
    return {
        'available': True,
        'files_found': list(gene_sets.keys()),
        'n_files': len(gene_sets),
        'per_file_gene_counts': {k: len(v) for k, v in gene_sets.items()},
        'union_n_genes': len(union),
        'intersection_n_genes': len(inter),
        'union_genes': union,
        'intersection_genes': inter,
    }
