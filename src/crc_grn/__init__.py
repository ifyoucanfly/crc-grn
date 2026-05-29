from .pipeline import run_demo_pipeline, run_real_pipeline, audit_project_inputs
from .ccc_model import (
    infer_spatial_lr_edges,
    build_ccc_signatures,
    associate_driver_ccc,
    build_ie_pathways,
)
from .preprocessing import normalize_expression_matrix, robust_gene_filter, samplewise_rank_transform
from .metrics import edge_prediction_metrics, load_gold_edges

# MODIFIED: bump package version to identify the v3 output-driven optimization build.
__version__ = '0.4.0-output-driven-v3'
# END MODIFIED
