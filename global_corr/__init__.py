"""
Global correspondence matching using SIFT + Fisher Vectors.
"""

from .global_matching import (
    build_fisher_vectors,
    compute_fisher_vector,
    compute_fv_pair_scores,
    extract_sift_features,
    import_and_match_pairs,
    main,
    read_descriptors_from_database,
    run_global_matching,
    save_pairs_to_file,
    select_top_pairs_per_image,
    train_gmm_diagonal,
)

__all__ = [
    "extract_sift_features",
    "read_descriptors_from_database",
    "train_gmm_diagonal",
    "build_fisher_vectors",
    "compute_fisher_vector",
    "compute_fv_pair_scores",
    "select_top_pairs_per_image",
    "save_pairs_to_file",
    "import_and_match_pairs",
    "run_global_matching",
    "main",
]
