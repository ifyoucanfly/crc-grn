#!/usr/bin/env python
from pathlib import Path
import yaml

root = Path(__file__).resolve().parents[1]
with open(root / 'config' / 'samples.yaml', 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)

main_root = root / 'data' / 'main_visiumhd'
sc_root = root / 'data' / 'sc_reference'
prior_root = root / 'resources' / 'grn_prior'
demo_root = root / 'data' / 'demo_bundle'
xenium_root = root / 'data' / 'orthogonal_xenium_gse280314'
external_root = root / 'data' / 'external_validation_gse226997'
extension_root = root / 'data' / 'extension_gse267401'


def exists_any(paths):
    return any(Path(p).exists() for p in paths)

print('[demo]')
for p in [demo_root / 'demo_expr.csv', demo_root / 'demo_meta.csv']:
    print(f'{p.name}: {p.exists()}')

print('\n[real main_visiumhd]')
for sample, gsm in cfg['sample_to_gsm'].items():
    h5 = exists_any([main_root / f'{sample}_filtered_feature_bc_matrix.h5', main_root / f'{gsm}_{sample}_filtered_feature_bc_matrix.h5'])
    pos = exists_any([
        main_root / f'{sample}_tissue_positions.parquet.gz',
        main_root / f'{sample}_tissue_positions.parquet',
        main_root / f'{sample}_tissue_positions.csv.gz',
        main_root / f'{sample}_tissue_positions.csv',
        main_root / f'{gsm}_{sample}_tissue_positions.parquet.gz',
        main_root / f'{gsm}_{sample}_tissue_positions.parquet',
        main_root / f'{gsm}_{sample}_tissue_positions.csv.gz',
        main_root / f'{gsm}_{sample}_tissue_positions.csv',
    ])
    meta = exists_any([
        main_root / f'{sample}_Metadata.parquet.gz',
        main_root / f'{sample}_Metadata.parquet',
        main_root / f'{gsm}_{sample}_Metadata.parquet.gz',
        main_root / f'{gsm}_{sample}_Metadata.parquet',
    ])
    print(sample, {'h5': h5, 'positions': pos, 'metadata': meta})

print('\n[sc_reference optional]')
for name in [
    'GSE132465_GEO_processed_CRC_10X_cell_annotation.txt.gz',
    'GSE132465_GEO_processed_CRC_10X_raw_UMI_count_matrix.txt.gz',
    'GSE144735_processed_KUL3_CRC_10X_annotation.txt.gz',
    'GSE144735_processed_KUL3_CRC_10X_raw_UMI_count_matrix.txt.gz',
]:
    print(name, (sc_root / name).exists())

print('\n[priors]')
for name in ['collectri_human.tsv', 'trrust_human.tsv']:
    print(name, (prior_root / name).exists())

print('\n[orthogonal_xenium_gse280314]')
for name in [
    'P1_CRC_outs.zip',
    'P2_CRC_outs.zip',
    'P5_CRC_outs.zip',
    'gene_panel.json.gz',
    'GSE280314_gene_panel.json.gz',
]:
    print(name, (xenium_root / name).exists())

print('\n[external_validation_gse226997]')
if external_root.exists():
    items = sorted([p for p in external_root.iterdir() if p.is_dir() or p.name.endswith('.tar.gz') or p.name.endswith('.tgz') or p.suffix == '.zip'])
    for sub in items:
        if sub.is_dir():
            flags = {
                'barcodes': exists_any(list(sub.glob('*barcodes.tsv*'))),
                'features': exists_any(list(sub.glob('*features.tsv*'))),
                'matrix': exists_any(list(sub.glob('*matrix.mtx*'))),
                'positions': exists_any(list(sub.glob('*tissue_positions*')) + list(sub.glob('*positions_list*'))),
                'source_type': 'directory',
            }
        else:
            flags = {'archive': True, 'source_type': 'archive'}
        print(sub.name, flags)
else:
    print('folder missing')

print('\n[extension_gse267401]')
if extension_root.exists():
    for sub in sorted([p for p in extension_root.iterdir() if p.is_dir()]):
        flags = {
            'barcodes': exists_any(list(sub.glob('*barcodes.tsv*'))),
            'features': exists_any(list(sub.glob('*features.tsv*'))),
            'matrix': exists_any(list(sub.glob('*matrix.mtx*'))),
            'positions': exists_any(list(sub.glob('*tissue_positions*'))),
        }
        print(sub.name, flags)
else:
    print('folder missing')
