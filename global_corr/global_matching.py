#!/usr/bin/env python3
"""
Global correspondence matching using SIFT + Fisher Vectors + COLMAP CLI.

This script performs efficient image pair matching by:
1. Extracting SIFT features using COLMAP CLI with GPU
2. Training a GMM and encoding Fisher Vectors per image
3. Finding top-k neighbors using Fisher Vector cosine similarity
4. Selecting candidate pairs based on similarity scores
5. Performing geometric verification using COLMAP CLI
"""

import argparse
import os
import shutil
import sqlite3
import subprocess
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from loguru import logger
from tqdm import tqdm

DEFAULT_GMM_BATCH_SIZE = 2048
DEFAULT_FV_BATCH_SIZE = 1024
DEFAULT_SIM_BLOCK_SIZE = 256
COLMAP_MAX_IMAGE_ID = 2147483647


def _require_colmap() -> str:
    colmap_path = shutil.which("colmap")
    if colmap_path is None:
        raise RuntimeError("COLMAP executable not found in PATH")
    return colmap_path


def _pair_id_to_image_ids(pair_id: int) -> Tuple[int, int]:
    # See COLMAP src/colmap/scene/database.cc (Database::PairIdToImagePair).
    image_id2 = pair_id % COLMAP_MAX_IMAGE_ID
    image_id1 = (pair_id - image_id2) // COLMAP_MAX_IMAGE_ID
    return int(image_id1), int(image_id2)


def _get_verified_pair_stats(database_path: str) -> Dict[int, Tuple[int, int]]:
    """
    Return per-image stats from `two_view_geometries`:
      image_id -> (num_verified_pairs_incident, total_inliers_incident)

    Note: `two_view_geometries.data` stores inlier matches as uint32 pairs.
    """
    stats: Dict[int, Tuple[int, int]] = defaultdict(lambda: (0, 0))
    conn = sqlite3.connect(database_path)
    cur = conn.cursor()
    cur.execute(
        "SELECT pair_id, data FROM two_view_geometries "
        "WHERE data IS NOT NULL AND length(data) > 0"
    )
    for pair_id, blob in cur.fetchall():
        if blob is None:
            continue
        # each match = 2 uint32 => 8 bytes
        num_inliers = len(blob) // 8
        i1, i2 = _pair_id_to_image_ids(int(pair_id))

        c1, s1 = stats[i1]
        stats[i1] = (c1 + 1, s1 + num_inliers)
        c2, s2 = stats[i2]
        stats[i2] = (c2 + 1, s2 + num_inliers)
    conn.close()
    return dict(stats)


def _get_existing_pair_ids(database_path: str) -> Set[int]:
    """
    Return set of pair_ids already present in the database.
    Prefer `matches` (should align with two_view_geometries), but union both.
    """
    conn = sqlite3.connect(database_path)
    cur = conn.cursor()
    pair_ids: Set[int] = set()
    for table in ("matches", "two_view_geometries"):
        try:
            cur.execute(f"SELECT pair_id FROM {table}")
            pair_ids.update(int(r[0]) for r in cur.fetchall())
        except sqlite3.OperationalError:
            # table might not exist in some DB variants
            continue
    conn.close()
    return pair_ids


def _expand_pairs_for_weak_images(
    pair_scores: Dict[Tuple[int, int], float],
    weak_image_ids: Set[int],
    existing_pairs: Set[Tuple[int, int]],
    extra_pairs_per_image: int,
) -> List[Tuple[int, int]]:
    """
    Add up to `extra_pairs_per_image` new pairs incident to each weak image,
    taking highest-FV-similarity edges not already in `existing_pairs`.
    """
    if extra_pairs_per_image <= 0 or not weak_image_ids:
        return []

    neighbors: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
    for (i, j), score in pair_scores.items():
        if i in weak_image_ids:
            neighbors[i].append((j, score))
        if j in weak_image_ids:
            neighbors[j].append((i, score))

    for i in neighbors:
        neighbors[i].sort(key=lambda x: x[1], reverse=True)

    new_pairs: Set[Tuple[int, int]] = set()
    for i in sorted(weak_image_ids):
        added = 0
        for j, _score in neighbors.get(i, []):
            if added >= extra_pairs_per_image:
                break
            pair = (i, j) if i < j else (j, i)
            if pair in existing_pairs or pair in new_pairs:
                continue
            new_pairs.add(pair)
            added += 1

    return sorted(new_pairs)


class _ImageUnionFind:
    def __init__(self, elements: Set[int]):
        self.parent: Dict[int, int] = {int(e): int(e) for e in elements}
        self.rank: Dict[int, int] = {int(e): 0 for e in elements}

    def find(self, x: int) -> int:
        x = int(x)
        p = self.parent[x]
        if p != x:
            self.parent[x] = self.find(p)
        return self.parent[x]

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True


def _compute_mst_edges(
    pair_scores: Dict[Tuple[int, int], float],
    image_ids_set: Set[int],
) -> List[Tuple[int, int]]:
    """
    Compute a maximum-spanning forest over images using FV similarity.

    We use only edges present in `pair_scores` (usually top-k neighbors per image),
    so if the FV-neighbor graph is disconnected this returns a forest.
    """
    if not image_ids_set or not pair_scores:
        return []

    edges = sorted(pair_scores.items(), key=lambda kv: kv[1], reverse=True)
    uf = _ImageUnionFind(set(int(x) for x in image_ids_set))

    mst_edges: List[Tuple[int, int]] = []
    for (i, j), _score in edges:
        i = int(i)
        j = int(j)
        if uf.union(i, j):
            mst_edges.append((i, j) if i < j else (j, i))
            if len(mst_edges) >= max(0, len(image_ids_set) - 1):
                break

    roots = {uf.find(int(i)) for i in image_ids_set}
    if len(roots) > 1:
        logger.info(
            "FV-MST produced %d components (forest edges=%d). "
            "If you want stronger connectivity, increase --k_neighbors.",
            len(roots),
            len(mst_edges),
        )
    else:
        logger.info("FV-MST produced a connected spanning tree (edges=%d).", len(mst_edges))

    return mst_edges


def _add_top_edges_per_image(
    pair_scores: Dict[Tuple[int, int], float],
    image_ids_set: Set[int],
    existing_pairs: Set[Tuple[int, int]],
    extra_edges_per_image: int,
) -> List[Tuple[int, int]]:
    """
    Add up to `extra_edges_per_image` highest-score edges per image.

    This is a lightweight densification on top of an MST/forest (or any initial
    pair set). It uses only edges present in `pair_scores` (typically top-k FV
    neighbors), so it stays far from exhaustive.
    """
    if extra_edges_per_image <= 0 or not image_ids_set or not pair_scores:
        return []

    neighbors: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
    for (i, j), score in pair_scores.items():
        i = int(i)
        j = int(j)
        if i in image_ids_set:
            neighbors[i].append((j, score))
        if j in image_ids_set:
            neighbors[j].append((i, score))

    for i in neighbors:
        neighbors[i].sort(key=lambda x: x[1], reverse=True)

    new_pairs: Set[Tuple[int, int]] = set()
    for i in sorted(image_ids_set):
        added = 0
        for j, _score in neighbors.get(int(i), []):
            if added >= extra_edges_per_image:
                break
            pair = (int(i), int(j)) if int(i) < int(j) else (int(j), int(i))
            if pair in existing_pairs or pair in new_pairs:
                continue
            new_pairs.add(pair)
            added += 1

    return sorted(new_pairs)


def extract_sift_features(
    database_path: str,
    image_dir: str,
    use_gpu: bool = True,
    max_num_features: int = 8192,
    max_image_size: int = 3200,
    camera_model: str = "SIMPLE_RADIAL",
    num_threads: int = -1,
    gpu_index: str = "0",
) -> None:
    """
    Extract SIFT features using COLMAP CLI with GPU support.

    Args:
        database_path: Path to the COLMAP database
        image_dir: Directory containing images
        use_gpu: Whether to use GPU for feature extraction
        max_num_features: Maximum number of features per image
        max_image_size: Maximum image size (longest side) for SIFT
        camera_model: COLMAP camera model used by the ImageReader
        num_threads: Number of threads (-1: auto)
        gpu_index: GPU index string (e.g. "0" or "0,1")
    """
    logger.info(f"Extracting SIFT features from {image_dir}")

    # Find colmap executable
    colmap_path = _require_colmap()

    # Build the feature_extractor command
    cmd = [
        colmap_path,
        "feature_extractor",
        "--database_path", database_path,
        "--image_path", image_dir,
        "--ImageReader.camera_model", camera_model,
        "--SiftExtraction.num_threads", str(num_threads),
        "--SiftExtraction.max_num_features", str(max_num_features),
        "--SiftExtraction.max_image_size", str(max_image_size),
        "--SiftExtraction.use_gpu", "1" if use_gpu else "0",
    ]

    if use_gpu:
        cmd.extend(["--SiftExtraction.gpu_index", gpu_index])

    logger.info(f"Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=False)

    if result.returncode != 0:
        raise RuntimeError(f"COLMAP feature_extractor failed with code {result.returncode}")

    logger.info("Feature extraction complete")


def read_descriptors_from_database(
    database_path: str,
) -> Tuple[np.ndarray, np.ndarray, Dict[int, str], np.ndarray]:
    """
    Read all descriptors from COLMAP database.

    Returns:
        descriptors: (N, 128) array of SIFT descriptors
        image_ids: (N,) array mapping each descriptor to its image ID
        image_id_to_name: Dict mapping image ID to image name
        keypoint_counts: Number of keypoints per image (for offset calculation)
    """
    logger.info(f"Reading descriptors from {database_path}")

    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()

    # Get all images
    cursor.execute("SELECT image_id, name FROM images")
    image_rows = cursor.fetchall()
    image_id_to_name = {row[0]: row[1] for row in image_rows}

    # Collect all descriptors with their image IDs
    all_descriptors = []
    all_image_ids = []
    keypoint_counts = []

    for image_id, image_name in tqdm(image_rows, desc="Reading descriptors"):
        cursor.execute(
            "SELECT data FROM descriptors WHERE image_id = ?", (image_id,)
        )
        row = cursor.fetchone()
        if row is None or row[0] is None:
            keypoint_counts.append(0)
            continue

        # Descriptors are stored as blob in COLMAP format
        blob = row[0]
        # COLMAP stores descriptors as uint8 with shape (num_features, 128)
        descriptors = np.frombuffer(blob, dtype=np.uint8).reshape(-1, 128)
        num_features = descriptors.shape[0]

        all_descriptors.append(descriptors)
        all_image_ids.extend([image_id] * num_features)
        keypoint_counts.append(num_features)

    conn.close()

    if len(all_descriptors) == 0:
        raise RuntimeError("No descriptors found in database")

    descriptors = np.vstack(all_descriptors).astype(np.float32)
    image_ids = np.array(all_image_ids, dtype=np.int64)

    logger.info(
        f"Loaded {len(descriptors)} descriptors from {len(image_id_to_name)} images"
    )

    return descriptors, image_ids, image_id_to_name, np.array(keypoint_counts)


def _get_device() -> torch.device:
    """Get the best available device (CUDA if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _compute_log_prob_diag_gmm_torch(
    descriptors: torch.Tensor,
    weights: torch.Tensor,
    means: torch.Tensor,
    variances: torch.Tensor,
    eps: float = 1e-10,
) -> torch.Tensor:
    """
    Compute log-probabilities of descriptors under a diagonal GMM using PyTorch.

    Args:
        descriptors: (N, D) tensor of descriptors
        weights: (K,) tensor of mixture weights
        means: (K, D) tensor of component means
        variances: (K, D) tensor of diagonal covariances
        eps: Small constant for numerical stability

    Returns:
        (N, K) tensor of log-probabilities
    """
    log_weights = torch.log(weights + eps)
    log_det = torch.sum(torch.log(2.0 * torch.pi * variances), dim=1)
    # (N, 1, D) - (1, K, D) -> (N, K, D)
    diff = descriptors.unsqueeze(1) - means.unsqueeze(0)
    mahal = torch.sum((diff * diff) / variances.unsqueeze(0), dim=2)
    return log_weights.unsqueeze(0) - 0.5 * (log_det.unsqueeze(0) + mahal)


def _compute_responsibilities_torch(
    descriptors: torch.Tensor,
    weights: torch.Tensor,
    means: torch.Tensor,
    variances: torch.Tensor,
    eps: float = 1e-10,
) -> torch.Tensor:
    """
    Compute soft assignments (responsibilities) for descriptors under a diagonal GMM.
    Uses log-sum-exp trick for numerical stability.

    Args:
        descriptors: (N, D) tensor of descriptors
        weights: (K,) tensor of mixture weights
        means: (K, D) tensor of component means
        variances: (K, D) tensor of diagonal covariances
        eps: Small constant for numerical stability

    Returns:
        (N, K) tensor of responsibilities (soft assignments)
    """
    log_prob = _compute_log_prob_diag_gmm_torch(descriptors, weights, means, variances, eps=eps)
    # Use torch's built-in log_softmax for numerical stability
    log_resp = log_prob - torch.logsumexp(log_prob, dim=1, keepdim=True)
    return torch.exp(log_resp)


def train_gmm_diagonal(
    descriptors: np.ndarray,
    n_components: int = 64,
    n_iters: int = 20,
    sample_size: int = 100000,
    min_covar: float = 1e-6,
    seed: int = 0,
    batch_size: int = DEFAULT_GMM_BATCH_SIZE,
) -> Dict[str, np.ndarray]:
    """
    Train a diagonal-covariance GMM using EM on a random descriptor subset.
    Uses PyTorch for GPU acceleration.

    Args:
        descriptors: (N, D) array of descriptors
        n_components: Number of GMM components
        n_iters: Number of EM iterations
        sample_size: Number of descriptors to sample for training
        min_covar: Minimum variance (regularization)
        seed: Random seed for reproducibility
        batch_size: Batch size for EM updates

    Returns:
        Dictionary with 'weights', 'means', 'variances' as numpy arrays
    """
    device = _get_device()
    logger.info(f"Training GMM on device: {device}")

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    descriptors = descriptors.astype(np.float32, copy=False)
    num_desc = descriptors.shape[0]

    if num_desc == 0:
        raise RuntimeError("No descriptors available to train GMM")

    # Sample subset for training
    if sample_size < num_desc:
        sample_idx = rng.choice(num_desc, size=sample_size, replace=False)
        sample_np = descriptors[sample_idx]
    else:
        sample_np = descriptors

    num_samples, dim = sample_np.shape
    if num_samples < n_components:
        logger.warning(
            "Reducing GMM components from %d to %d due to limited samples",
            n_components,
            num_samples,
        )
        n_components = max(1, num_samples)

    # Move sample data to GPU
    sample = torch.from_numpy(sample_np).to(device)

    # Initialize GMM parameters on GPU
    init_idx = rng.choice(num_samples, size=n_components, replace=False)
    means = sample[init_idx].clone()
    variances = sample.var(dim=0, keepdim=True) + min_covar
    variances = variances.repeat(n_components, 1)
    weights = torch.full((n_components,), 1.0 / n_components, device=device, dtype=torch.float32)

    logger.info(
        "Training diagonal GMM with %d components on %d samples",
        n_components,
        num_samples,
    )

    for iteration in range(n_iters):
        # Accumulators for sufficient statistics
        sum_resp = torch.zeros(n_components, device=device, dtype=torch.float32)
        sum_resp_x = torch.zeros((n_components, dim), device=device, dtype=torch.float32)
        sum_resp_x2 = torch.zeros((n_components, dim), device=device, dtype=torch.float32)

        # E-step: compute responsibilities in batches
        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            batch = sample[start:end]

            resp = _compute_responsibilities_torch(batch, weights, means, variances)

            # Accumulate sufficient statistics
            sum_resp += resp.sum(dim=0)
            # (K, N) @ (N, D) -> (K, D)
            sum_resp_x += resp.T @ batch
            sum_resp_x2 += resp.T @ (batch * batch)

        # M-step: update parameters
        weights = sum_resp / (sum_resp.sum() + 1e-10)
        means = sum_resp_x / (sum_resp.unsqueeze(1) + 1e-10)
        variances = sum_resp_x2 / (sum_resp.unsqueeze(1) + 1e-10) - means * means
        variances = torch.clamp(variances, min=min_covar)

        logger.info("GMM iteration %d/%d complete", iteration + 1, n_iters)

    # Move results back to CPU as numpy arrays
    return {
        "weights": weights.cpu().numpy(),
        "means": means.cpu().numpy(),
        "variances": variances.cpu().numpy(),
    }


def compute_fisher_vector(
    descriptors: np.ndarray,
    gmm: Dict[str, np.ndarray],
    batch_size: int = DEFAULT_FV_BATCH_SIZE,
    eps: float = 1e-10,
) -> np.ndarray:
    """
    Compute a Fisher Vector for a set of descriptors under a diagonal GMM.
    Uses PyTorch for GPU acceleration.

    The Fisher Vector encodes first and second-order statistics of how
    the descriptors deviate from the GMM parameters.

    Args:
        descriptors: (N, D) array of descriptors
        gmm: Dictionary with 'weights', 'means', 'variances'
        batch_size: Batch size for computation
        eps: Small constant for numerical stability

    Returns:
        (2 * K * D,) Fisher Vector (power-normalized and L2-normalized)
    """
    device = _get_device()

    weights_np = gmm["weights"]
    means_np = gmm["means"]
    variances_np = gmm["variances"]
    num_components, dim = means_np.shape

    if descriptors.size == 0:
        return np.zeros(2 * num_components * dim, dtype=np.float32)

    descriptors = descriptors.astype(np.float32, copy=False)
    num_desc = descriptors.shape[0]

    # Move GMM parameters to GPU
    weights = torch.from_numpy(weights_np).to(device)
    means = torch.from_numpy(means_np).to(device)
    variances = torch.from_numpy(variances_np).to(device)

    # Precompute normalization terms
    inv_sigma = 1.0 / torch.sqrt(variances + eps)
    inv_sigma2 = 1.0 / (variances + eps)
    sqrt_weights = torch.sqrt(weights + eps)
    sqrt_2_weights = torch.sqrt(2.0 * weights + eps)

    # Accumulators for Fisher Vector components
    sum_u = torch.zeros((num_components, dim), device=device, dtype=torch.float32)
    sum_v = torch.zeros((num_components, dim), device=device, dtype=torch.float32)

    # Process in batches
    desc_tensor = torch.from_numpy(descriptors).to(device)

    for start in range(0, num_desc, batch_size):
        end = min(start + batch_size, num_desc)
        batch = desc_tensor[start:end]

        # Compute responsibilities: (batch_size, K)
        resp = _compute_responsibilities_torch(batch, weights, means, variances, eps=eps)

        # Compute deviations: (batch_size, K, D)
        diff = batch.unsqueeze(1) - means.unsqueeze(0)

        # First-order (gradient w.r.t. means): weighted normalized deviations
        # (batch_size, K, D) weighted by (batch_size, K, 1)
        weighted_diff = resp.unsqueeze(2) * diff * inv_sigma.unsqueeze(0)
        sum_u += weighted_diff.sum(dim=0)

        # Second-order (gradient w.r.t. variances): weighted squared deviations - 1
        weighted_sq = resp.unsqueeze(2) * ((diff * diff) * inv_sigma2.unsqueeze(0) - 1.0)
        sum_v += weighted_sq.sum(dim=0)

    # Normalize by number of descriptors and weight terms
    sum_u = sum_u / (num_desc * sqrt_weights.unsqueeze(1) + eps)
    sum_v = sum_v / (num_desc * sqrt_2_weights.unsqueeze(1) + eps)

    # Concatenate first and second order terms
    fv = torch.cat([sum_u, sum_v], dim=0).reshape(-1)

    # Power normalization (signed square root)
    fv = torch.sign(fv) * torch.sqrt(torch.abs(fv) + eps)

    # L2 normalization
    fv = fv / (torch.linalg.norm(fv) + eps)

    return fv.cpu().numpy().astype(np.float32)


def _build_image_slices(image_ids: np.ndarray) -> Dict[int, Tuple[int, int]]:
    """
    Build contiguous descriptor slices for each image id.
    """
    if image_ids.size == 0:
        return {}

    change_idx = np.where(np.diff(image_ids) != 0)[0] + 1
    starts = np.concatenate(([0], change_idx))
    ends = np.concatenate((change_idx, [len(image_ids)]))
    return {
        int(image_ids[start]): (int(start), int(end))
        for start, end in zip(starts, ends)
    }


def build_fisher_vectors(
    descriptors: np.ndarray,
    image_ids: np.ndarray,
    image_id_to_name: Dict[int, str],
    gmm: Dict[str, np.ndarray],
    fv_batch_size: int = DEFAULT_FV_BATCH_SIZE,
) -> Tuple[List[int], np.ndarray]:
    """
    Encode Fisher Vectors for all images in the database.
    """
    image_ids_sorted = sorted(image_id_to_name.keys())
    num_components, dim = gmm["means"].shape
    fv_dim = 2 * num_components * dim
    fv_matrix = np.zeros((len(image_ids_sorted), fv_dim), dtype=np.float32)
    image_slices = _build_image_slices(image_ids)

    for row_idx, image_id in enumerate(
        tqdm(image_ids_sorted, desc="Encoding Fisher vectors")
    ):
        if image_id not in image_slices:
            continue
        start, end = image_slices[image_id]
        fv_matrix[row_idx] = compute_fisher_vector(
            descriptors[start:end], gmm, batch_size=fv_batch_size
        )

    return image_ids_sorted, fv_matrix


def compute_fv_pair_scores(
    fv_matrix: np.ndarray,
    image_ids: List[int],
    neighbors_per_image: int = 50,
    block_size: int = DEFAULT_SIM_BLOCK_SIZE,
) -> Dict[Tuple[int, int], float]:
    """
    Compute cosine-similarity scores for top neighbors per image.
    """
    num_images = fv_matrix.shape[0]
    if num_images == 0:
        return {}

    k = min(neighbors_per_image, max(0, num_images - 1))
    if k == 0:
        return {}

    norms = np.linalg.norm(fv_matrix, axis=1, keepdims=True)
    valid_mask = norms.squeeze(axis=1) > 1e-6
    fv_normed = np.zeros_like(fv_matrix)
    fv_normed[valid_mask] = fv_matrix[valid_mask] / norms[valid_mask]

    logger.info(
        "Computing FV cosine similarities (neighbors_per_image=%d)", neighbors_per_image
    )
    pair_scores: Dict[Tuple[int, int], float] = {}

    for start in tqdm(range(0, num_images, block_size), desc="FV similarity blocks"):
        end = min(start + block_size, num_images)
        sims = fv_normed[start:end] @ fv_normed.T
        if not np.all(valid_mask):
            sims[:, ~valid_mask] = -np.inf
        row_ids = np.arange(start, end)
        sims[np.arange(end - start), row_ids] = -np.inf

        top_idx = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
        top_sims = np.take_along_axis(sims, top_idx, axis=1)
        order = np.argsort(-top_sims, axis=1)
        top_idx = np.take_along_axis(top_idx, order, axis=1)
        top_sims = np.take_along_axis(top_sims, order, axis=1)

        for row_offset, (neighbor_indices, neighbor_sims) in enumerate(
            zip(top_idx, top_sims)
        ):
            if not valid_mask[start + row_offset]:
                continue
            img_i = image_ids[start + row_offset]
            for neighbor_idx, sim in zip(neighbor_indices, neighbor_sims):
                if neighbor_idx < 0 or np.isneginf(sim) or np.isnan(sim):
                    continue
                img_j = image_ids[neighbor_idx]
                if img_i == img_j:
                    continue
                pair = (img_i, img_j) if img_i < img_j else (img_j, img_i)
                current = pair_scores.get(pair)
                if current is None or sim > current:
                    pair_scores[pair] = float(sim)

    logger.info(f"Found {len(pair_scores)} unique candidate pairs")
    return pair_scores


def select_top_pairs_per_image(
    pair_scores: Dict[Tuple[int, int], float],
    image_ids_set: Set[int],
    top_k_pairs: int = 50,
) -> List[Tuple[int, int]]:
    """
    Select top-k highest-scoring pairs for each image, avoiding duplicates.

    For each image, we select the top-k neighbors based on score.
    If (A, B) is selected, (B, A) won't be added again.

    Args:
        pair_scores: Dict mapping (img_id_1, img_id_2) -> score
        image_ids_set: Set of all image IDs
        top_k_pairs: Number of pairs to select per image

    Returns:
        selected_pairs: List of (img_id_1, img_id_2) pairs
    """
    logger.info(f"Selecting top {top_k_pairs} pairs per image...")

    # Build per-image neighbor lists with frequencies
    image_neighbors: Dict[int, List[Tuple[int, int]]] = defaultdict(list)

    for (img1, img2), score in pair_scores.items():
        image_neighbors[img1].append((img2, score))
        image_neighbors[img2].append((img1, score))

    # Sort neighbors by frequency (descending) for each image
    for img_id in image_neighbors:
        image_neighbors[img_id].sort(key=lambda x: x[1], reverse=True)

    # Select pairs greedily, avoiding duplicates
    selected_pairs_set: Set[Tuple[int, int]] = set()

    for img_id in sorted(image_ids_set):
        neighbors = image_neighbors.get(img_id, [])
        count = 0

        for neighbor_id, freq in neighbors:
            if count >= top_k_pairs:
                break

            # Canonical ordering
            if img_id < neighbor_id:
                pair = (img_id, neighbor_id)
            else:
                pair = (neighbor_id, img_id)

            # Only add if not already in set
            if pair not in selected_pairs_set:
                selected_pairs_set.add(pair)
                count += 1

    selected_pairs = sorted(selected_pairs_set)
    logger.info(f"Selected {len(selected_pairs)} unique pairs")

    return selected_pairs


def save_pairs_to_file(
    pairs: List[Tuple[int, int]],
    image_id_to_name: Dict[int, str],
    output_path: str,
) -> None:
    """
    Save image pairs to a text file in COLMAP format.

    Args:
        pairs: List of (img_id_1, img_id_2) pairs
        image_id_to_name: Dict mapping image ID to image name
        output_path: Path to output file
    """
    logger.info(f"Saving {len(pairs)} pairs to {output_path}")

    with open(output_path, "w") as f:
        for img1, img2 in pairs:
            name1 = image_id_to_name[img1]
            name2 = image_id_to_name[img2]
            f.write(f"{name1} {name2}\n")

    logger.info("Pairs saved successfully")


def import_and_match_pairs(
    database_path: str,
    pairs_path: str,
    use_gpu: bool = True,
    num_threads: int = -1,
    gpu_index: str = "0",
    guided_matching: bool = False,
    max_num_matches: int = 32768,
    max_ratio: float = 0.8,
    max_distance: float = 0.7,
    cross_check: bool = True,
    cpu_brute_force_matcher: bool = False,
    min_num_inliers: int = 15,
    max_error: float = 4.0,
    confidence: float = 0.999,
    max_num_trials: int = 10000,
    min_inlier_ratio: float = 0.25,
) -> None:
    """
    Import pairs and run feature matching + geometric verification using COLMAP.

    This calls `colmap matches_importer` which performs both matching and
    geometric verification for the specified pairs.

    Args:
        database_path: Path to COLMAP database
        pairs_path: Path to pairs file (image name pairs, one per line)
        use_gpu: Whether to use GPU for matching
    """
    logger.info("Running COLMAP matches_importer...")

    # Find colmap executable
    colmap_path = _require_colmap()

    cmd = [
        colmap_path,
        "matches_importer",
        "--database_path", database_path,
        "--match_list_path", pairs_path,
        "--match_type", "pairs",
        "--SiftMatching.num_threads", str(num_threads),
        "--SiftMatching.use_gpu", "1" if use_gpu else "0",
        "--SiftMatching.guided_matching", "1" if guided_matching else "0",
        "--SiftMatching.max_num_matches", str(max_num_matches),
        "--SiftMatching.max_ratio", str(max_ratio),
        "--SiftMatching.max_distance", str(max_distance),
        "--SiftMatching.cross_check", "1" if cross_check else "0",
        # "--SiftMatching.cpu_brute_force_matcher",
        # "1" if cpu_brute_force_matcher else "0",
        "--TwoViewGeometry.min_num_inliers", str(min_num_inliers),
        "--TwoViewGeometry.max_error", str(max_error),
        "--TwoViewGeometry.confidence", str(confidence),
        "--TwoViewGeometry.max_num_trials", str(max_num_trials),
        "--TwoViewGeometry.min_inlier_ratio", str(min_inlier_ratio),
    ]

    if use_gpu:
        cmd.extend(["--SiftMatching.gpu_index", gpu_index])

    logger.info(f"Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=False)

    if result.returncode != 0:
        raise RuntimeError(f"COLMAP matches_importer failed with code {result.returncode}")

    logger.info("Matching and verification complete")


def run_exhaustive_matcher(
    database_path: str,
    use_gpu: bool = True,
    num_threads: int = -1,
    gpu_index: str = "0",
    guided_matching: bool = False,
    max_num_matches: int = 32768,
    max_ratio: float = 0.8,
    max_distance: float = 0.7,
    cross_check: bool = True,
    cpu_brute_force_matcher: bool = False,
    min_num_inliers: int = 15,
    max_error: float = 4.0,
    confidence: float = 0.999,
    max_num_trials: int = 10000,
    min_inlier_ratio: float = 0.25,
    block_size: int = 50,
) -> None:
    """
    Run COLMAP exhaustive_matcher on an existing database.

    Use this to build an apples-to-apples baseline: same features + same matching
    / verification options, but enumerating all pairs.
    """
    logger.info("Running COLMAP exhaustive_matcher...")

    colmap_path = _require_colmap()
    cmd = [
        colmap_path,
        "exhaustive_matcher",
        "--database_path", database_path,
        "--FeatureMatching.num_threads", str(num_threads),
        "--FeatureMatching.use_gpu", "1" if use_gpu else "0",
        "--FeatureMatching.guided_matching", "1" if guided_matching else "0",
        "--FeatureMatching.max_num_matches", str(max_num_matches),
        "--SiftMatching.max_ratio", str(max_ratio),
        "--SiftMatching.max_distance", str(max_distance),
        "--SiftMatching.cross_check", "1" if cross_check else "0",
        "--SiftMatching.cpu_brute_force_matcher",
        "1" if cpu_brute_force_matcher else "0",
        "--TwoViewGeometry.min_num_inliers", str(min_num_inliers),
        "--TwoViewGeometry.max_error", str(max_error),
        "--TwoViewGeometry.confidence", str(confidence),
        "--TwoViewGeometry.max_num_trials", str(max_num_trials),
        "--TwoViewGeometry.min_inlier_ratio", str(min_inlier_ratio),
        "--ExhaustiveMatching.block_size", str(block_size),
    ]
    if use_gpu:
        cmd.extend(["--FeatureMatching.gpu_index", gpu_index])

    logger.info(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"COLMAP exhaustive_matcher failed with code {result.returncode}"
        )


def clone_database_without_matches(src_database_path: str, dst_database_path: str) -> None:
    """
    Clone a COLMAP database, removing any matches / two-view geometries.

    This is useful when you want to run multiple matchers on the same features.
    """
    if os.path.abspath(src_database_path) == os.path.abspath(dst_database_path):
        raise ValueError("src_database_path and dst_database_path must be different")
    shutil.copy2(src_database_path, dst_database_path)
    conn = sqlite3.connect(dst_database_path)
    cur = conn.cursor()
    for table in ("matches", "two_view_geometries"):
        cur.execute(f"DELETE FROM {table}")
    conn.commit()
    conn.close()


def run_global_matching(
    image_dir: str,
    output_dir: str,
    k_neighbors: int = 50,
    top_k_pairs: int = 50,
    max_num_features: int = 8192,
    max_image_size: int = 3200,
    gmm_components: int = 64,
    gmm_iters: int = 20,
    gmm_sample_size: int = 100000,
    gmm_min_covar: float = 1e-6,
    gmm_seed: int = 0,
    use_gpu: bool = True,
    gpu_index: str = "0",
    num_threads: int = -1,
    guided_matching: bool = False,
    max_num_matches: int = 32768,
    max_ratio: float = 0.8,
    max_distance: float = 0.7,
    cross_check: bool = True,
    cpu_brute_force_matcher: bool = False,
    min_num_inliers: int = 15,
    two_view_max_error: float = 4.0,
    two_view_confidence: float = 0.999,
    two_view_max_num_trials: int = 10000,
    two_view_min_inlier_ratio: float = 0.25,
    add_mst_edges: bool = False,
    extra_edges_per_image: int = 0,
    adaptive_pair_expansion: bool = False,
    stage2_weak_fraction: float = 0.25,
    stage2_extra_pairs_per_image: int = 20,
    run_exhaustive_baseline: bool = False,
    exhaustive_block_size: int = 50,
) -> str:
    """
    Run the complete global matching pipeline.

    Args:
        image_dir: Directory containing images
        output_dir: Directory for output files
        k_neighbors: Candidate neighbors per image for FV similarity
        top_k_pairs: Number of pairs to select per image
        max_num_features: Maximum SIFT features per image
        max_image_size: Maximum image size (longest side) for SIFT
        gmm_components: Number of GMM components for Fisher Vectors
        gmm_iters: Number of EM iterations for the GMM
        gmm_sample_size: Number of descriptors sampled to train the GMM
        gmm_min_covar: Variance floor for the GMM
        gmm_seed: Random seed for GMM sampling
        use_gpu: Whether to use GPU
        gpu_index: GPU index string for COLMAP (e.g. "0" or "0,1")
        num_threads: Number of CPU threads for COLMAP matchers (-1: auto)
        guided_matching: Enable guided matching (slower, can improve inliers)
        max_num_matches: Cap number of matches per image pair
        max_ratio: SIFT ratio test threshold
        max_distance: SIFT distance threshold
        cross_check: Enable cross-check filtering
        cpu_brute_force_matcher: Force CPU brute-force matcher
        min_num_inliers: Minimum inliers for two-view geometry to be accepted
        two_view_max_error: RANSAC max error (pixels)
        two_view_confidence: RANSAC confidence
        two_view_max_num_trials: RANSAC max trials
        two_view_min_inlier_ratio: Minimum inlier ratio for early termination
        add_mst_edges: Add FV-MST edges to improve connectivity cheaply
        extra_edges_per_image: Add a few extra high-FV edges per image (cheap densification)
        adaptive_pair_expansion: Two-stage matching; add extra pairs for weak images
        stage2_weak_fraction: Fraction of weakest images to expand (0..1]
        stage2_extra_pairs_per_image: Extra pairs added per weak image in stage 2
        run_exhaustive_baseline: Also run exhaustive matcher on same features
        exhaustive_block_size: Exhaustive matcher block size

    Returns:
        Path to the pairs file
    """
    os.makedirs(output_dir, exist_ok=True)

    database_path = os.path.join(output_dir, "database.db")
    pairs_path = os.path.join(output_dir, "pairs.txt")
    stage2_pairs_path = os.path.join(output_dir, "pairs_stage2.txt")
    exhaustive_database_path = os.path.join(output_dir, "database_exhaustive.db")

    # Step 1: Extract SIFT features
    logger.info("=" * 60)
    logger.info("Step 1: SIFT Feature Extraction")
    logger.info("=" * 60)

    # Database will be created automatically by extract_features if it doesn't exist
    extract_sift_features(
        database_path=database_path,
        image_dir=image_dir,
        use_gpu=use_gpu,
        max_num_features=max_num_features,
        max_image_size=max_image_size,
        num_threads=num_threads,
        gpu_index=gpu_index,
    )

    if run_exhaustive_baseline:
        logger.info("=" * 60)
        logger.info("Baseline: cloning database for exhaustive matching")
        logger.info("=" * 60)
        clone_database_without_matches(
            src_database_path=database_path,
            dst_database_path=exhaustive_database_path,
        )

    # Step 2: Read descriptors from database
    logger.info("=" * 60)
    logger.info("Step 2: Reading Descriptors")
    logger.info("=" * 60)

    descriptors, image_ids, image_id_to_name, _ = read_descriptors_from_database(
        database_path
    )

    # Step 3: Train GMM and encode Fisher Vectors
    logger.info("=" * 60)
    logger.info("Step 3: Training GMM and Encoding Fisher Vectors")
    logger.info("=" * 60)

    gmm = train_gmm_diagonal(
        descriptors=descriptors,
        n_components=gmm_components,
        n_iters=gmm_iters,
        sample_size=gmm_sample_size,
        min_covar=gmm_min_covar,
        seed=gmm_seed,
    )
    image_ids_sorted, fv_matrix = build_fisher_vectors(
        descriptors=descriptors,
        image_ids=image_ids,
        image_id_to_name=image_id_to_name,
        gmm=gmm,
    )

    # Step 4: Compute Fisher Vector similarity scores
    logger.info("=" * 60)
    logger.info("Step 4: Computing FV Pair Scores")
    logger.info("=" * 60)

    neighbors_per_image = max(
        k_neighbors,
        top_k_pairs + (stage2_extra_pairs_per_image if adaptive_pair_expansion else 0),
    )
    if neighbors_per_image != top_k_pairs or neighbors_per_image != k_neighbors:
        logger.info(
            "Using neighbors_per_image=%d (k_neighbors=%d, top_k_pairs=%d)",
            neighbors_per_image,
            k_neighbors,
            top_k_pairs,
        )
    pair_scores = compute_fv_pair_scores(
        fv_matrix=fv_matrix,
        image_ids=image_ids_sorted,
        neighbors_per_image=neighbors_per_image,
    )

    # Step 5: Select top pairs per image
    logger.info("=" * 60)
    logger.info("Step 5: Selecting Top Pairs")
    logger.info("=" * 60)

    image_ids_set = set(image_id_to_name.keys())
    selected_pairs = select_top_pairs_per_image(
        pair_scores=pair_scores,
        image_ids_set=image_ids_set,
        top_k_pairs=top_k_pairs,
    )

    if add_mst_edges:
        logger.info("=" * 60)
        logger.info("Step 5b: Adding FV-MST edges (connectivity boost)")
        logger.info("=" * 60)
        mst_edges = _compute_mst_edges(pair_scores=pair_scores, image_ids_set=image_ids_set)
        before = len(selected_pairs)
        selected_pairs_set = set(selected_pairs)
        selected_pairs_set.update(mst_edges)
        selected_pairs = sorted(selected_pairs_set)
        logger.info(
            "Added %d FV-MST edges (pairs: %d -> %d)",
            len(mst_edges),
            before,
            len(selected_pairs),
        )

        if extra_edges_per_image > 0:
            logger.info("=" * 60)
            logger.info("Step 5c: Adding extra FV edges per image (densification)")
            logger.info("=" * 60)
            selected_pairs_set = set(selected_pairs)
            extra_edges = _add_top_edges_per_image(
                pair_scores=pair_scores,
                image_ids_set=image_ids_set,
                existing_pairs=selected_pairs_set,
                extra_edges_per_image=extra_edges_per_image,
            )
            before2 = len(selected_pairs_set)
            selected_pairs_set.update(extra_edges)
            selected_pairs = sorted(selected_pairs_set)
            logger.info(
                "Added %d extra edges (pairs: %d -> %d, extra_edges_per_image=%d)",
                len(extra_edges),
                before2,
                len(selected_pairs),
                extra_edges_per_image,
            )

    # Step 6: Save pairs to file
    logger.info("=" * 60)
    logger.info("Step 6: Saving Pairs")
    logger.info("=" * 60)

    save_pairs_to_file(
        pairs=selected_pairs,
        image_id_to_name=image_id_to_name,
        output_path=pairs_path,
    )

    # Step 7: Feature matching and geometric verification
    logger.info("=" * 60)
    logger.info("Step 7: Feature Matching & Geometric Verification")
    logger.info("=" * 60)

    import_and_match_pairs(
        database_path=database_path,
        pairs_path=pairs_path,
        use_gpu=use_gpu,
        num_threads=num_threads,
        gpu_index=gpu_index,
        guided_matching=guided_matching,
        max_num_matches=max_num_matches,
        max_ratio=max_ratio,
        max_distance=max_distance,
        cross_check=cross_check,
        cpu_brute_force_matcher=cpu_brute_force_matcher,
        min_num_inliers=min_num_inliers,
        max_error=two_view_max_error,
        confidence=two_view_confidence,
        max_num_trials=two_view_max_num_trials,
        min_inlier_ratio=two_view_min_inlier_ratio,
    )

    # Optional Stage 2: expand pairs only for weak images, then match only new pairs.
    if adaptive_pair_expansion:
        logger.info("=" * 60)
        logger.info("Stage 8: Adaptive pair expansion (stage-2)")
        logger.info("=" * 60)

        if not (0.0 < stage2_weak_fraction <= 1.0):
            raise ValueError("--stage2_weak_fraction must be in (0, 1]")

        stats = _get_verified_pair_stats(database_path)
        # Missing images (no verified pairs) are maximally weak.
        per_image = []
        for image_id in image_ids_set:
            deg, inliers = stats.get(int(image_id), (0, 0))
            per_image.append((deg, inliers, int(image_id)))
        per_image.sort(key=lambda x: (x[0], x[1], x[2]))  # weakest first

        num_weak = max(1, int(round(stage2_weak_fraction * len(per_image))))
        weak_image_ids = {img_id for _deg, _inl, img_id in per_image[:num_weak]}
        logger.info(
            "Expanding pairs for %d/%d weakest images (fraction=%.3f), extra_pairs_per_image=%d",
            len(weak_image_ids),
            len(per_image),
            stage2_weak_fraction,
            stage2_extra_pairs_per_image,
        )

        existing_pairs_set = set(selected_pairs)
        extra_pairs = _expand_pairs_for_weak_images(
            pair_scores=pair_scores,
            weak_image_ids=weak_image_ids,
            existing_pairs=existing_pairs_set,
            extra_pairs_per_image=stage2_extra_pairs_per_image,
        )
        all_pairs_set = existing_pairs_set.union(extra_pairs)
        logger.info(
            "Stage-2 selected %d extra pairs (total pairs now %d)",
            len(extra_pairs),
            len(all_pairs_set),
        )

        # Filter out any pairs already present in the DB (paranoia against reruns).
        existing_pair_ids = _get_existing_pair_ids(database_path)
        filtered_extra_pairs: List[Tuple[int, int]] = []
        for i, j in extra_pairs:
            pair_id = i * COLMAP_MAX_IMAGE_ID + j
            if pair_id in existing_pair_ids:
                continue
            filtered_extra_pairs.append((i, j))

        if not filtered_extra_pairs:
            logger.info("No new pairs to match in stage-2 (all already present).")
        else:
            save_pairs_to_file(
                pairs=filtered_extra_pairs,
                image_id_to_name=image_id_to_name,
                output_path=stage2_pairs_path,
            )
            import_and_match_pairs(
                database_path=database_path,
                pairs_path=stage2_pairs_path,
                use_gpu=use_gpu,
                num_threads=num_threads,
                gpu_index=gpu_index,
                guided_matching=guided_matching,
                max_num_matches=max_num_matches,
                max_ratio=max_ratio,
                max_distance=max_distance,
                cross_check=cross_check,
                cpu_brute_force_matcher=cpu_brute_force_matcher,
                min_num_inliers=min_num_inliers,
                max_error=two_view_max_error,
                confidence=two_view_confidence,
                max_num_trials=two_view_max_num_trials,
                min_inlier_ratio=two_view_min_inlier_ratio,
            )

        # Keep pairs.txt as the full list we intended to cover.
        save_pairs_to_file(
            pairs=sorted(all_pairs_set),
            image_id_to_name=image_id_to_name,
            output_path=pairs_path,
        )

    if run_exhaustive_baseline:
        logger.info("=" * 60)
        logger.info("Baseline: exhaustive matching on cloned database")
        logger.info("=" * 60)
        run_exhaustive_matcher(
            database_path=exhaustive_database_path,
            use_gpu=use_gpu,
            num_threads=num_threads,
            gpu_index=gpu_index,
            guided_matching=guided_matching,
            max_num_matches=max_num_matches,
            max_ratio=max_ratio,
            max_distance=max_distance,
            cross_check=cross_check,
            cpu_brute_force_matcher=cpu_brute_force_matcher,
            min_num_inliers=min_num_inliers,
            max_error=two_view_max_error,
            confidence=two_view_confidence,
            max_num_trials=two_view_max_num_trials,
            min_inlier_ratio=two_view_min_inlier_ratio,
            block_size=exhaustive_block_size,
        )

    logger.info("=" * 60)
    logger.info("Global matching pipeline complete!")
    logger.info(f"Database: {database_path}")
    logger.info(f"Pairs: {pairs_path}")
    if adaptive_pair_expansion:
        logger.info(f"Stage-2 pairs (new): {stage2_pairs_path}")
    if run_exhaustive_baseline:
        logger.info(f"Exhaustive DB: {exhaustive_database_path}")
    logger.info("=" * 60)

    return pairs_path


def main():
    parser = argparse.ArgumentParser(
        description="Global correspondence matching using SIFT + Fisher Vectors"
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="Directory containing input images",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory for output files (database, pairs, etc.)",
    )
    parser.add_argument(
        "--k_neighbors",
        type=int,
        default=50,
        help=(
            "Candidate neighbors per image for FV similarity "
            "(default: 50, max with --top_k_pairs)"
        ),
    )
    parser.add_argument(
        "--top_k_pairs",
        type=int,
        default=50,
        help="Number of image pairs to select per image (default: 50)",
    )
    parser.add_argument(
        "--max_num_features",
        type=int,
        default=8192,
        help="Maximum SIFT features per image (default: 8192)",
    )
    parser.add_argument(
        "--max_image_size",
        type=int,
        default=3200,
        help="Maximum image size (longest side) for SIFT (default: 3200)",
    )
    parser.add_argument(
        "--gmm_components",
        type=int,
        default=64,
        help="Number of GMM components for Fisher Vectors (default: 64)",
    )
    parser.add_argument(
        "--gmm_iters",
        type=int,
        default=20,
        help="Number of EM iterations for GMM training (default: 20)",
    )
    parser.add_argument(
        "--gmm_sample_size",
        type=int,
        default=100000,
        help="Number of descriptors sampled to train the GMM (default: 100000)",
    )
    parser.add_argument(
        "--gmm_min_covar",
        type=float,
        default=1e-6,
        help="Variance floor for GMM training (default: 1e-6)",
    )
    parser.add_argument(
        "--gmm_seed",
        type=int,
        default=0,
        help="Random seed for GMM sampling (default: 0)",
    )
    parser.add_argument(
        "--no_gpu",
        action="store_true",
        help="Disable GPU acceleration",
    )
    parser.add_argument(
        "--gpu_index",
        type=str,
        default="0",
        help='GPU index for COLMAP (default: "0")',
    )
    parser.add_argument(
        "--num_threads",
        type=int,
        default=-1,
        help="Number of CPU threads for COLMAP (-1: auto)",
    )
    parser.add_argument(
        "--no_guided_matching",
        action="store_false",
        dest="guided_matching",
        help="Disable guided matching in COLMAP matchers (enabled by default)",
    )
    parser.add_argument(
        "--max_num_matches",
        type=int,
        default=32768,
        help="Max matches per image pair (default: 32768)",
    )
    parser.add_argument(
        "--sift_max_ratio",
        type=float,
        default=0.8,
        help="SIFT ratio test threshold (default: 0.8)",
    )
    parser.add_argument(
        "--sift_max_distance",
        type=float,
        default=0.7,
        help="SIFT max distance threshold (default: 0.7)",
    )
    parser.add_argument(
        "--no_cross_check",
        action="store_true",
        help="Disable SIFT cross-check (enabled by default)",
    )
    parser.add_argument(
        "--cpu_brute_force_matcher",
        action="store_true",
        help="Force CPU brute force matcher for SIFT matching",
    )
    parser.add_argument(
        "--min_num_inliers",
        type=int,
        default=15,
        help="Two-view geometry min inliers (default: 15)",
    )
    parser.add_argument(
        "--two_view_max_error",
        type=float,
        default=4.0,
        help="Two-view geometry max error in pixels (default: 4.0)",
    )
    parser.add_argument(
        "--two_view_confidence",
        type=float,
        default=0.999,
        help="Two-view geometry RANSAC confidence (default: 0.999)",
    )
    parser.add_argument(
        "--two_view_max_num_trials",
        type=int,
        default=10000,
        help="Two-view geometry max RANSAC trials (default: 10000)",
    )
    parser.add_argument(
        "--two_view_min_inlier_ratio",
        type=float,
        default=0.25,
        help="Two-view geometry min inlier ratio (default: 0.25)",
    )
    parser.add_argument(
        "--no_mst_edges",
        action="store_false",
        dest="add_mst_edges",
        help=(
            "Disable FV-based maximum-spanning forest edges "
            "(enabled by default for connectivity)"
        ),
    )
    parser.add_argument(
        "--extra_edges_per_image",
        type=int,
        default=3,
        help=(
            "When used with --add_mst_edges, add up to this many extra high-FV "
            "edges per image for cheap densification (default: 3)"
        ),
    )
    parser.add_argument(
        "--adaptive_pair_expansion",
        action="store_true",
        help=(
            "Enable 2-stage matching: start with top_k_pairs, then add a small "
            "number of extra pairs for weak images and match only those new pairs"
        ),
    )
    parser.add_argument(
        "--stage2_weak_fraction",
        type=float,
        default=0.25,
        help="Fraction of weakest images to expand in stage-2 (default: 0.25)",
    )
    parser.add_argument(
        "--stage2_extra_pairs_per_image",
        type=int,
        default=20,
        help="Extra pairs added per weak image in stage-2 (default: 20)",
    )
    parser.add_argument(
        "--run_exhaustive_baseline",
        action="store_true",
        help="Also run COLMAP exhaustive_matcher on same features",
    )
    parser.add_argument(
        "--exhaustive_block_size",
        type=int,
        default=50,
        help="ExhaustiveMatching.block_size (default: 50)",
    )
    parser.add_argument(
        "--database",
        type=str,
        default=None,
        help="Path to existing COLMAP database (skip feature extraction)",
    )

    args = parser.parse_args()

    # Run pipeline
    if args.database is not None:
        # Use existing database - skip extraction, just do matching
        logger.info(f"Using existing database: {args.database}")

        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        pairs_path = os.path.join(output_dir, "pairs.txt")

        # Read descriptors
        descriptors, image_ids, image_id_to_name, _ = read_descriptors_from_database(
            args.database
        )

        # Train GMM, encode Fisher Vectors, and find pairs
        use_gpu = not args.no_gpu
        cross_check = not args.no_cross_check
        gmm = train_gmm_diagonal(
            descriptors=descriptors,
            n_components=args.gmm_components,
            n_iters=args.gmm_iters,
            sample_size=args.gmm_sample_size,
            min_covar=args.gmm_min_covar,
            seed=args.gmm_seed,
        )
        image_ids_sorted, fv_matrix = build_fisher_vectors(
            descriptors=descriptors,
            image_ids=image_ids,
            image_id_to_name=image_id_to_name,
            gmm=gmm,
        )

        neighbors_per_image = max(args.k_neighbors, args.top_k_pairs)
        if neighbors_per_image != args.top_k_pairs or neighbors_per_image != args.k_neighbors:
            logger.info(
                "Using neighbors_per_image=%d (k_neighbors=%d, top_k_pairs=%d)",
                neighbors_per_image,
                args.k_neighbors,
                args.top_k_pairs,
            )
        pair_scores = compute_fv_pair_scores(
            fv_matrix=fv_matrix,
            image_ids=image_ids_sorted,
            neighbors_per_image=neighbors_per_image,
        )

        image_ids_set = set(image_id_to_name.keys())
        selected_pairs = select_top_pairs_per_image(
            pair_scores=pair_scores,
            image_ids_set=image_ids_set,
            top_k_pairs=args.top_k_pairs,
        )

        if args.add_mst_edges:
            mst_edges = _compute_mst_edges(pair_scores=pair_scores, image_ids_set=image_ids_set)
            selected_pairs_set = set(selected_pairs)
            selected_pairs_set.update(mst_edges)
            selected_pairs = sorted(selected_pairs_set)
            logger.info(
                "Added %d FV-MST edges (total pairs=%d)",
                len(mst_edges),
                len(selected_pairs),
            )
            if args.extra_edges_per_image > 0:
                selected_pairs_set = set(selected_pairs)
                extra_edges = _add_top_edges_per_image(
                    pair_scores=pair_scores,
                    image_ids_set=image_ids_set,
                    existing_pairs=selected_pairs_set,
                    extra_edges_per_image=args.extra_edges_per_image,
                )
                selected_pairs_set.update(extra_edges)
                selected_pairs = sorted(selected_pairs_set)
                logger.info(
                    "Added %d extra edges (total pairs=%d, extra_edges_per_image=%d)",
                    len(extra_edges),
                    len(selected_pairs),
                    args.extra_edges_per_image,
                )

        save_pairs_to_file(
            pairs=selected_pairs,
            image_id_to_name=image_id_to_name,
            output_path=pairs_path,
        )

        # Match and verify
        import_and_match_pairs(
            database_path=args.database,
            pairs_path=pairs_path,
            use_gpu=use_gpu,
            num_threads=args.num_threads,
            gpu_index=args.gpu_index,
            guided_matching=args.guided_matching,
            max_num_matches=args.max_num_matches,
            max_ratio=args.sift_max_ratio,
            max_distance=args.sift_max_distance,
            cross_check=cross_check,
            cpu_brute_force_matcher=args.cpu_brute_force_matcher,
            min_num_inliers=args.min_num_inliers,
            max_error=args.two_view_max_error,
            confidence=args.two_view_confidence,
            max_num_trials=args.two_view_max_num_trials,
            min_inlier_ratio=args.two_view_min_inlier_ratio,
        )

    else:
        # Full pipeline
        run_global_matching(
            image_dir=args.image_dir,
            output_dir=args.output_dir,
            k_neighbors=args.k_neighbors,
            top_k_pairs=args.top_k_pairs,
            max_num_features=args.max_num_features,
            max_image_size=args.max_image_size,
            gmm_components=args.gmm_components,
            gmm_iters=args.gmm_iters,
            gmm_sample_size=args.gmm_sample_size,
            gmm_min_covar=args.gmm_min_covar,
            gmm_seed=args.gmm_seed,
            use_gpu=not args.no_gpu,
            gpu_index=args.gpu_index,
            num_threads=args.num_threads,
            guided_matching=args.guided_matching,
            max_num_matches=args.max_num_matches,
            max_ratio=args.sift_max_ratio,
            max_distance=args.sift_max_distance,
            cross_check=not args.no_cross_check,
            cpu_brute_force_matcher=args.cpu_brute_force_matcher,
            min_num_inliers=args.min_num_inliers,
            two_view_max_error=args.two_view_max_error,
            two_view_confidence=args.two_view_confidence,
            two_view_max_num_trials=args.two_view_max_num_trials,
            two_view_min_inlier_ratio=args.two_view_min_inlier_ratio,
            add_mst_edges=args.add_mst_edges,
            extra_edges_per_image=args.extra_edges_per_image,
            adaptive_pair_expansion=args.adaptive_pair_expansion,
            stage2_weak_fraction=args.stage2_weak_fraction,
            stage2_extra_pairs_per_image=args.stage2_extra_pairs_per_image,
            run_exhaustive_baseline=args.run_exhaustive_baseline,
            exhaustive_block_size=args.exhaustive_block_size,
        )


if __name__ == "__main__":
    main()
