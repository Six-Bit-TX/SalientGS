#!/usr/bin/env python3
"""
Evaluate camera pose accuracy on ETH3D SLAM benchmark.

Based on the COLMAP benchmark reconstruction evaluation protocol
(colmap/benchmark/reconstruction/evaluation/utils.py).

Key features:
  - Uses COLMAP model_aligner for robust Sim3 alignment (LORANSAC + Umeyama)
  - Absolute pose error evaluation (translation in meters)
  - Relative pairwise pose error evaluation (angular distance in degrees)
  - AUC and Recall metrics matching COLMAP benchmark exactly
  - Correct denominator: number of dataset images with GT (from rgb.txt)
  - Supports joint-optimized pose evaluation via gsplat checkpoints

Usage:
    # Single scene (SfM only)
    python eval_eth3d_slam_poses.py \\
        --scene_dir /path/to/scene \\
        --sparse_dir /path/to/sparse/0 \\
        --output results.json

    # Single scene (joint optimized)
    python eval_eth3d_slam_poses.py \\
        --scene_dir /path/to/scene \\
        --sparse_dir /path/to/sparse/0 \\
        --ckpt /path/to/checkpoint.pt \\
        --output results.json

    # Batch (all scenes)
    python eval_eth3d_slam_poses.py \\
        --batch_dir /path/to/workdir \\
        --output results.json
"""

import argparse
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

try:
    import pycolmap
except ImportError:
    pycolmap = None

# NumPy compat: trapezoid was added in NumPy 2.0
_trapz = getattr(np, "trapezoid", None) or np.trapz

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

POSITION_ACCURACY_GT = 0.01  # meters – min_error for AUC computation
ALIGNMENT_MAX_ERROR = 0.2  # meters – RANSAC inlier threshold for Sim3
ABS_THRESHOLDS = np.array([0.02, 0.05, 0.1, 0.2, 0.5])
REL_THRESHOLDS = np.array([0.5, 1.0, 5.0, 10.0])
DEFAULT_COLMAP = "/usr/local/bin/colmap"

SCENE_PREFIXES = [
    "cables", "camera", "ceiling", "desk", "einstein", "kidnap",
    "large", "mannequin", "motion", "planar", "plant", "reflective",
    "repetitive", "sfm", "sofa", "table", "vicon",
]


# ---------------------------------------------------------------------------
# Quaternion / rotation utilities
# ---------------------------------------------------------------------------

def quat_to_rot(qw, qx, qy, qz):
    n = qw * qw + qx * qx + qy * qy + qz * qz
    s = 2.0 / max(n, 1e-12)
    wx, wy, wz = s * qw * qx, s * qw * qy, s * qw * qz
    xx, xy, xz = s * qx * qx, s * qx * qy, s * qx * qz
    yy, yz, zz = s * qy * qy, s * qy * qz, s * qz * qz
    return np.array([
        [1 - (yy + zz), xy - wz, xz + wy],
        [xy + wz, 1 - (xx + zz), yz - wx],
        [xz - wy, yz + wx, 1 - (xx + yy)],
    ])


def rot_to_quat(R):
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (R[2, 1] - R[1, 2]) * s
        qy = (R[0, 2] - R[2, 0]) * s
        qz = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    return qw, qx, qy, qz


# ---------------------------------------------------------------------------
# TUM ground-truth loading
# ---------------------------------------------------------------------------

def parse_tum_groundtruth(gt_path: Path) -> dict[str, np.ndarray]:
    """Parse TUM-format groundtruth.txt -> {timestamp_str: 4x4 c2w matrix}."""
    poses = {}
    with open(gt_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            ts = parts[0]
            tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
            qx, qy, qz, qw = (
                float(parts[4]), float(parts[5]),
                float(parts[6]), float(parts[7]),
            )
            R = quat_to_rot(qw, qx, qy, qz)
            c2w = np.eye(4)
            c2w[:3, :3] = R
            c2w[:3, 3] = [tx, ty, tz]
            poses[ts] = c2w
    return poses


def parse_rgb_txt(rgb_path: Path) -> dict[str, str]:
    """Parse rgb.txt -> {image_basename: timestamp_str}."""
    mapping = {}
    with open(rgb_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            ts = parts[0]
            img_basename = parts[1].split("/")[-1]
            mapping[img_basename] = ts
    return mapping


# ---------------------------------------------------------------------------
# Dataset image collection (determines evaluation denominator)
# ---------------------------------------------------------------------------

def collect_dataset_images(
    scene_dir: Path, max_dt: float = 0.05,
) -> list[tuple[str, np.ndarray]]:
    """Collect all dataset images that have a GT pose.

    Each entry in rgb.txt whose timestamp is within *max_dt* seconds of a GT
    timestamp is considered a "dataset image". The length of this list is the
    denominator for recall/AUC – analogous to len(sparse_gt.images) in the
    COLMAP benchmark.

    Returns list of (image_basename, gt_c2w_4x4).
    """
    gt_poses = parse_tum_groundtruth(scene_dir / "groundtruth.txt")
    if not (scene_dir / "rgb.txt").exists():
        return []

    img_to_ts = parse_rgb_txt(scene_dir / "rgb.txt")
    gt_timestamps = sorted(gt_poses.keys(), key=float)
    gt_ts_float = np.array([float(t) for t in gt_timestamps])

    dataset_images = []
    for img_name, ts_str in img_to_ts.items():
        ts_f = float(ts_str)
        idx = np.argmin(np.abs(gt_ts_float - ts_f))
        if abs(gt_ts_float[idx] - ts_f) <= max_dt:
            dataset_images.append((img_name, gt_poses[gt_timestamps[idx]]))
    return dataset_images


# ---------------------------------------------------------------------------
# COLMAP reconstruction helpers
# ---------------------------------------------------------------------------

def _get_recon_image_map(recon) -> dict[str, object]:
    """Build basename -> pycolmap.Image mapping."""
    images = {}
    for img in recon.images.values():
        basename = img.name.split("/")[-1]
        images[basename] = img
    return images


def _image_center(img) -> np.ndarray:
    """Get projection center (camera position in world coords)."""
    R = img.cam_from_world.rotation.matrix()
    t = img.cam_from_world.translation
    return -R.T @ t


def _image_c2w(img) -> np.ndarray:
    """Get 4x4 camera-to-world matrix from a pycolmap Image."""
    R = img.cam_from_world.rotation.matrix()
    t = img.cam_from_world.translation
    c2w = np.eye(4)
    c2w[:3, :3] = R.T
    c2w[:3, 3] = -R.T @ t
    return c2w


# ---------------------------------------------------------------------------
# Checkpoint pose-delta application
# ---------------------------------------------------------------------------

def _get_colmap_image_order(sparse_dir: Path) -> list[str]:
    """Image names ordered by COLMAP image_id (matches gsplat CameraOptModule)."""
    assert pycolmap is not None
    recon = pycolmap.Reconstruction(str(sparse_dir))
    return [
        recon.images[k].name.split("/")[-1]
        for k in sorted(recon.images.keys())
    ]


def _rotation_6d_to_matrix(d6: np.ndarray) -> np.ndarray:
    a1 = d6[:3]
    a2 = d6[3:6]
    b1 = a1 / max(np.linalg.norm(a1), 1e-8)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / max(np.linalg.norm(b2), 1e-8)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)


def apply_checkpoint_deltas(
    recon, sparse_dir: Path, ckpt_path: Path,
):
    """Apply CameraOptModule deltas and write a modified reconstruction.

    Returns a new pycolmap.Reconstruction with adjusted poses.
    """
    import torch

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if "pose_adjust" not in ckpt:
        print(f"  Warning: no pose_adjust in {ckpt_path}, using base poses.")
        return recon

    embeds = ckpt["pose_adjust"]["embeds.weight"].numpy()
    identity_6d = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    image_order = _get_colmap_image_order(sparse_dir)

    name_to_delta = {}
    for idx, name in enumerate(image_order):
        if idx >= len(embeds):
            break
        delta = embeds[idx]
        dx = delta[:3]
        drot_6d = delta[3:] + identity_6d
        rot = _rotation_6d_to_matrix(drot_6d)
        T = np.eye(4)
        T[:3, :3] = rot
        T[:3, 3] = dx
        name_to_delta[name] = T

    img_map = _get_recon_image_map(recon)
    for basename, delta_T in name_to_delta.items():
        img = img_map.get(basename)
        if img is None:
            continue
        base_c2w = _image_c2w(img)
        new_c2w = base_c2w @ delta_T
        new_w2c = np.linalg.inv(new_c2w)
        R_w2c = new_w2c[:3, :3]
        t_w2c = new_w2c[:3, 3]
        img.cam_from_world = pycolmap.Rigid3d(
            pycolmap.Rotation3d(R_w2c), t_w2c
        )

    return recon


# ---------------------------------------------------------------------------
# Alignment via COLMAP model_aligner
# ---------------------------------------------------------------------------

def write_reference_file(
    dataset_images: list[tuple[str, np.ndarray]],
    recon,
    ref_path: Path,
) -> int:
    """Write GT camera centers as reference positions for model_aligner.

    Only writes entries for images that are both in the reconstruction AND
    have a GT pose (i.e., are in dataset_images).

    Returns the number of reference entries written.
    """
    if recon is None:
        return 0
    recon_names = {img.name.split("/")[-1]: img.name for img in recon.images.values()}
    gt_map = {name: c2w for name, c2w in dataset_images}

    count = 0
    with open(ref_path, "w") as f:
        for basename, full_name in recon_names.items():
            c2w = gt_map.get(basename)
            if c2w is None:
                continue
            center = c2w[:3, 3]
            f.write(f"{full_name} {center[0]:.10f} {center[1]:.10f} {center[2]:.10f}\n")
            count += 1
    return count


def run_model_aligner(
    colmap_path: str,
    input_path: Path,
    ref_path: Path,
    output_path: Path,
    max_error: float,
) -> bool:
    """Run COLMAP model_aligner. Returns True if alignment succeeded."""
    output_path.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(colmap_path), "model_aligner",
        "--input_path", str(input_path),
        "--output_path", str(output_path),
        "--ref_images_path", str(ref_path),
        "--ref_is_gps", "0",
        "--alignment_type", "custom",
        "--alignment_max_error", str(max_error),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  Warning: model_aligner returned {result.returncode}")
        print(f"  stderr: {result.stderr[:500]}")
        return False
    succeeded = "Alignment succeeded" in result.stderr
    if not succeeded:
        print("  Warning: model_aligner did not report success")
        for line in result.stderr.strip().split("\n"):
            if "error" in line.lower() or "align" in line.lower():
                print(f"    {line.strip()}")
    return succeeded


# ---------------------------------------------------------------------------
# Python Sim3 alignment fallback
# ---------------------------------------------------------------------------

def _umeyama_sim3(src: np.ndarray, dst: np.ndarray):
    """Umeyama Sim3 estimation: dst ≈ s * R @ src + t."""
    mu_s = src.mean(0)
    mu_d = dst.mean(0)
    sc = src - mu_s
    dc = dst - mu_d
    H = sc.T @ dc
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T
    scale = np.trace(R @ H) / np.trace(sc.T @ sc)
    t = mu_d - scale * R @ mu_s
    return scale, R, t


def _apply_sim3(pts, s, R, t):
    return s * (pts @ R.T) + t


def ransac_sim3(
    src: np.ndarray,
    dst: np.ndarray,
    max_error: float = 0.2,
    n_iter: int = 5000,
    rng=None,
):
    """RANSAC + Umeyama Sim3 with local refinement (LO-RANSAC style)."""
    if rng is None:
        rng = np.random.default_rng(42)
    n = len(src)
    if n < 3:
        if n == 0:
            return 1.0, np.eye(3), np.zeros(3)
        s, R, t = _umeyama_sim3(src, dst)
        return s, R, t

    best_n = 0
    best_s, best_R, best_t = 1.0, np.eye(3), np.zeros(3)

    for _ in range(n_iter):
        idx = rng.choice(n, 3, replace=False)
        s, R, t = _umeyama_sim3(src[idx], dst[idx])
        errs = np.linalg.norm(_apply_sim3(src, s, R, t) - dst, axis=1)
        inl = errs < max_error
        ni = inl.sum()
        if ni > best_n:
            best_n = ni
            best_s, best_R, best_t = s, R, t
            if ni == n:
                break

    # Local refinement on inlier set
    inliers = np.linalg.norm(
        _apply_sim3(src, best_s, best_R, best_t) - dst, axis=1
    ) < max_error
    if inliers.sum() >= 3:
        best_s, best_R, best_t = _umeyama_sim3(src[inliers], dst[inliers])
        # Second refinement
        inliers2 = np.linalg.norm(
            _apply_sim3(src, best_s, best_R, best_t) - dst, axis=1
        ) < max_error
        if inliers2.sum() > inliers.sum() and inliers2.sum() >= 3:
            best_s, best_R, best_t = _umeyama_sim3(src[inliers2], dst[inliers2])

    return best_s, best_R, best_t


def python_align_poses(
    dataset_images: list[tuple[str, np.ndarray]],
    recon,
    max_error: float = 0.2,
) -> dict[str, np.ndarray]:
    """Fallback alignment when COLMAP model_aligner is unavailable.

    Returns {image_basename: aligned_c2w_4x4}.
    """
    if recon is None:
        return {}

    img_map = _get_recon_image_map(recon)
    gt_map = {name: c2w for name, c2w in dataset_images}

    src, dst, names = [], [], []
    for basename, img in img_map.items():
        if basename not in gt_map:
            continue
        src.append(_image_center(img))
        dst.append(gt_map[basename][:3, 3])
        names.append(basename)

    if len(src) < 3:
        return {}

    src = np.array(src)
    dst = np.array(dst)
    s, R, t = ransac_sim3(src, dst, max_error=max_error)

    aligned = {}
    for basename, img in img_map.items():
        c2w = _image_c2w(img)
        c_old = c2w[:3, 3]
        c_new = s * (R @ c_old) + t
        R_new = R @ c2w[:3, :3]
        new_c2w = np.eye(4)
        new_c2w[:3, :3] = R_new
        new_c2w[:3, 3] = c_new
        aligned[basename] = new_c2w
    return aligned


# ---------------------------------------------------------------------------
# Error computation (matching COLMAP benchmark)
# ---------------------------------------------------------------------------

def compute_abs_errors(
    dataset_images: list[tuple[str, np.ndarray]],
    aligned_recon=None,
    aligned_poses: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    """Absolute per-image translation errors in meters.

    One error per dataset image. Unregistered images get infinity.
    Exactly matches colmap/benchmark/reconstruction/evaluation/utils.py
    compute_abs_errors().
    """
    n = len(dataset_images)
    errors = np.full(n, np.inf, dtype=np.float64)

    if aligned_recon is not None:
        img_map = _get_recon_image_map(aligned_recon)
        for i, (name, gt_c2w) in enumerate(dataset_images):
            img = img_map.get(name)
            if img is None:
                continue
            pred_center = _image_center(img)
            gt_center = gt_c2w[:3, 3]
            errors[i] = np.linalg.norm(pred_center - gt_center)
    elif aligned_poses is not None:
        for i, (name, gt_c2w) in enumerate(dataset_images):
            pred_c2w = aligned_poses.get(name)
            if pred_c2w is None:
                continue
            errors[i] = np.linalg.norm(pred_c2w[:3, 3] - gt_c2w[:3, 3])

    return errors


def _normalize_vec(v, eps=1e-10):
    return v / max(eps, float(np.linalg.norm(v)))


def _angular_dist_deg(v1, v2):
    cos = np.clip(np.dot(_normalize_vec(v1), _normalize_vec(v2)), -1, 1)
    return np.degrees(np.arccos(cos))


def _rotation_angle_deg(R):
    angle = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    return np.degrees(angle)


def compute_rel_errors(
    dataset_images: list[tuple[str, np.ndarray]],
    aligned_recon=None,
    aligned_poses: dict[str, np.ndarray] | None = None,
    min_proj_center_dist: float = 0.01,
) -> np.ndarray:
    """Relative pairwise pose errors in degrees.

    For each ordered pair (i, j), error = max(angular_translation_error,
    angular_rotation_error). Matches COLMAP benchmark compute_rel_errors().
    """
    n = len(dataset_images)
    pred_c2w_map: dict[str, np.ndarray] = {}

    if aligned_recon is not None:
        img_map = _get_recon_image_map(aligned_recon)
        for name, _ in dataset_images:
            img = img_map.get(name)
            if img is not None:
                pred_c2w_map[name] = _image_c2w(img)
    elif aligned_poses is not None:
        pred_c2w_map = aligned_poses

    errors = []
    for i in range(n):
        name_i, gt_c2w_i = dataset_images[i]
        if name_i not in pred_c2w_map:
            errors.extend([np.inf] * (n - 1))
            continue
        pred_w2c_i = np.linalg.inv(pred_c2w_map[name_i])
        gt_w2c_i = np.linalg.inv(gt_c2w_i)

        for j in range(n):
            if i == j:
                continue
            name_j, gt_c2w_j = dataset_images[j]
            if name_j not in pred_c2w_map:
                errors.append(np.inf)
                continue

            pred_w2c_j = np.linalg.inv(pred_c2w_map[name_j])
            gt_w2c_j = np.linalg.inv(gt_c2w_j)

            # Relative pose: j_from_i
            rel_pred = pred_w2c_j @ pred_c2w_map[name_i]
            rel_gt = gt_w2c_j @ gt_c2w_i

            # Rotation error
            est_from_gt = np.linalg.inv(rel_pred) @ rel_gt
            dR = _rotation_angle_deg(est_from_gt[:3, :3])

            # Translation error (angular)
            if np.linalg.norm(rel_gt[:3, 3]) < min_proj_center_dist:
                dt = 0.0
            else:
                dt = _angular_dist_deg(rel_pred[:3, 3], rel_gt[:3, 3])

            errors.append(max(dt, dR))

    return np.array(errors, dtype=np.float64)


# ---------------------------------------------------------------------------
# Metrics (exact copy of COLMAP benchmark)
# ---------------------------------------------------------------------------

def compute_auc(
    errors: np.ndarray,
    thresholds: np.ndarray,
    min_error: float = 0,
) -> np.ndarray:
    num_elems = len(errors)
    if num_elems == 0:
        return np.zeros(len(thresholds))

    errors = np.sort(errors)
    recalls = (np.arange(num_elems) + 1) / num_elems

    if min_error > 0:
        min_index = np.searchsorted(errors, min_error, side="right")
        min_recall = min_index / num_elems
        recalls = np.r_[min_recall, min_recall, recalls[min_index:]]
        errors = np.r_[0, min_error, errors[min_index:]]
    else:
        recalls = np.r_[0, recalls]
        errors = np.r_[0, errors]

    aucs = np.zeros(len(thresholds), dtype=np.float64)
    for i, t in enumerate(thresholds):
        last_index = np.searchsorted(errors, t, side="right")
        r = np.r_[recalls[:last_index], recalls[last_index - 1]]
        e = np.r_[errors[:last_index], t]
        aucs[i] = _trapz(r, x=e) / t * 100
    return aucs


def compute_recall(
    errors: np.ndarray,
    thresholds: np.ndarray,
) -> np.ndarray:
    num_elems = len(errors)
    if num_elems == 0:
        return np.zeros(len(thresholds))
    return np.array(
        [100.0 * np.sum(errors <= t) / num_elems for t in thresholds]
    )


# ---------------------------------------------------------------------------
# Single-scene evaluation
# ---------------------------------------------------------------------------

def evaluate_scene(
    scene_dir: Path,
    sparse_dir: Path,
    colmap_path: str = DEFAULT_COLMAP,
    error_type: str = "absolute",
    alignment_max_error: float = ALIGNMENT_MAX_ERROR,
    position_accuracy: float = POSITION_ACCURACY_GT,
    ckpt_path: Path | None = None,
) -> dict:
    """Evaluate a single ETH3D SLAM scene."""
    gt_path = scene_dir / "groundtruth.txt"
    if not gt_path.exists():
        return {"error": f"groundtruth.txt not found in {scene_dir}"}

    dataset_images = collect_dataset_images(scene_dir)
    n_dataset = len(dataset_images)
    if n_dataset == 0:
        return {"error": "no dataset images with GT", "num_images": 0}

    if not sparse_dir.exists():
        return _empty_result(n_dataset, 0, error_type, position_accuracy)

    if pycolmap is None:
        return {"error": "pycolmap is required"}

    recon = pycolmap.Reconstruction(str(sparse_dir))
    num_registered = recon.num_reg_images()

    if ckpt_path is not None and ckpt_path.exists():
        recon = apply_checkpoint_deltas(recon, sparse_dir, ckpt_path)

    # Count how many registered images have GT
    img_map = _get_recon_image_map(recon)
    gt_names = {name for name, _ in dataset_images}
    num_matched = sum(1 for b in img_map if b in gt_names)

    if num_matched < 3:
        return _empty_result(n_dataset, num_registered, error_type, position_accuracy)

    # Alignment + error computation
    thresholds = ABS_THRESHOLDS if error_type == "absolute" else REL_THRESHOLDS

    use_colmap = shutil.which(colmap_path) is not None
    if use_colmap:
        errors = _evaluate_with_colmap_aligner(
            dataset_images, recon, sparse_dir, colmap_path,
            alignment_max_error, error_type, position_accuracy,
            ckpt_path,
        )
    else:
        print(f"  Warning: COLMAP not found at {colmap_path}, using Python fallback")
        errors = _evaluate_with_python_aligner(
            dataset_images, recon, alignment_max_error, error_type,
            position_accuracy,
        )

    if errors is None:
        return _empty_result(n_dataset, num_registered, error_type, position_accuracy)

    aucs = compute_auc(errors, thresholds, min_error=position_accuracy)
    recalls = compute_recall(errors, thresholds)

    result = {
        "num_images": n_dataset,
        "num_registered": num_registered,
        "num_matched": num_matched,
        "error_type": error_type,
    }
    for i, t in enumerate(thresholds):
        unit = "m" if error_type == "absolute" else "deg"
        result[f"auc_{t}{unit}"] = float(aucs[i])
        result[f"recall_{t}{unit}"] = float(recalls[i])

    finite = errors[np.isfinite(errors)]
    if len(finite) > 0:
        result["median_error"] = float(np.median(finite))
        result["mean_error"] = float(np.mean(finite))

    return result


def _evaluate_with_colmap_aligner(
    dataset_images, recon, sparse_dir, colmap_path,
    alignment_max_error, error_type, position_accuracy,
    ckpt_path,
):
    """Align using COLMAP model_aligner and compute errors."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        ref_path = tmpdir / "ref.txt"

        n_ref = write_reference_file(dataset_images, recon, ref_path)
        if n_ref < 3:
            print(f"  Warning: only {n_ref} reference images, need >= 3")
            return None

        # If checkpoint was applied, write the modified reconstruction
        if ckpt_path is not None:
            input_dir = tmpdir / "modified_sparse"
            input_dir.mkdir()
            recon.write(str(input_dir))
        else:
            input_dir = sparse_dir

        aligned_dir = tmpdir / "aligned"
        ok = run_model_aligner(
            colmap_path, input_dir, ref_path, aligned_dir, alignment_max_error,
        )

        if not ok or not any(aligned_dir.glob("images.*")):
            print("  Warning: alignment failed, falling back to Python")
            return _evaluate_with_python_aligner(
                dataset_images, recon, alignment_max_error,
                error_type, position_accuracy,
            )

        aligned_recon = pycolmap.Reconstruction(str(aligned_dir))

        if error_type == "absolute":
            return compute_abs_errors(dataset_images, aligned_recon=aligned_recon)
        else:
            return compute_rel_errors(
                dataset_images, aligned_recon=aligned_recon,
                min_proj_center_dist=position_accuracy,
            )


def _evaluate_with_python_aligner(
    dataset_images, recon, alignment_max_error, error_type,
    position_accuracy,
):
    """Fallback: Python Sim3 alignment."""
    aligned_poses = python_align_poses(dataset_images, recon, alignment_max_error)
    if not aligned_poses:
        return None

    if error_type == "absolute":
        return compute_abs_errors(dataset_images, aligned_poses=aligned_poses)
    else:
        return compute_rel_errors(
            dataset_images, aligned_poses=aligned_poses,
            min_proj_center_dist=position_accuracy,
        )


def _empty_result(n_dataset, n_registered, error_type, position_accuracy):
    thresholds = ABS_THRESHOLDS if error_type == "absolute" else REL_THRESHOLDS
    unit = "m" if error_type == "absolute" else "deg"
    result = {
        "num_images": n_dataset,
        "num_registered": n_registered,
        "num_matched": 0,
        "error_type": error_type,
    }
    for t in thresholds:
        result[f"auc_{t}{unit}"] = 0.0
        result[f"recall_{t}{unit}"] = 0.0
    return result


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------

def get_scene_prefix(scene_name: str) -> str:
    for prefix in SCENE_PREFIXES:
        if scene_name.startswith(prefix):
            return prefix
    return scene_name


def _find_latest_ckpt(ckpt_dir: Path) -> Path | None:
    if not ckpt_dir.is_dir():
        return None
    ckpts = sorted(ckpt_dir.glob("ckpt_*_rank0.pt"))
    if not ckpts:
        return None
    ckpts.sort(key=lambda p: int(p.stem.split("_")[1]))
    return ckpts[-1]


def batch_evaluate(
    batch_dir: Path,
    sparse_subdir: str = "sparse/0",
    colmap_path: str = DEFAULT_COLMAP,
    error_type: str = "absolute",
    alignment_max_error: float = ALIGNMENT_MAX_ERROR,
    position_accuracy: float = POSITION_ACCURACY_GT,
    ckpt_subdir: str | None = None,
) -> dict:
    results = {}
    scene_dirs = sorted(
        d for d in batch_dir.iterdir()
        if d.is_dir() and (d / "groundtruth.txt").exists()
    )

    for scene_dir in scene_dirs:
        scene_name = scene_dir.name
        sparse_dir = scene_dir / sparse_subdir
        ckpt_path = (
            _find_latest_ckpt(scene_dir / ckpt_subdir)
            if ckpt_subdir else None
        )
        print(f"Evaluating {scene_name} ...")
        results[scene_name] = evaluate_scene(
            scene_dir, sparse_dir, colmap_path,
            error_type, alignment_max_error, position_accuracy,
            ckpt_path=ckpt_path,
        )
        r = results[scene_name]
        print(f"  images={r.get('num_images')}, registered={r.get('num_registered')}, "
              f"matched={r.get('num_matched')}")

    summary = _compute_summary(results, error_type)
    return {"per_scene": results, "summary": summary}


def _compute_summary(results: dict, error_type: str) -> dict:
    thresholds = ABS_THRESHOLDS if error_type == "absolute" else REL_THRESHOLDS
    unit = "m" if error_type == "absolute" else "deg"

    prefix_groups: dict[str, list[dict]] = {}
    for scene_name, metrics in results.items():
        prefix = get_scene_prefix(scene_name)
        prefix_groups.setdefault(prefix, []).append(metrics)

    summary = {}
    all_aucs = {f"{t}{unit}": [] for t in thresholds}
    all_recalls = {f"{t}{unit}": [] for t in thresholds}

    for prefix in SCENE_PREFIXES:
        group = prefix_groups.get(prefix, [])
        if not group:
            continue
        entry = {"n_scenes": len(group)}
        for t in thresholds:
            key = f"{t}{unit}"
            auc_val = np.mean([m.get(f"auc_{key}", 0) for m in group])
            rec_val = np.mean([m.get(f"recall_{key}", 0) for m in group])
            entry[f"auc_{key}"] = float(auc_val)
            entry[f"recall_{key}"] = float(rec_val)
            all_aucs[key].append(auc_val)
            all_recalls[key].append(rec_val)
        summary[prefix] = entry

    if all_aucs:
        avg = {}
        for t in thresholds:
            key = f"{t}{unit}"
            avg[f"auc_{key}"] = float(np.mean(all_aucs[key])) if all_aucs[key] else 0
            avg[f"recall_{key}"] = float(np.mean(all_recalls[key])) if all_recalls[key] else 0
        summary["average"] = avg

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate camera poses on ETH3D SLAM benchmark "
                    "(COLMAP benchmark protocol)"
    )
    parser.add_argument("--scene_dir", type=Path, default=None)
    parser.add_argument("--sparse_dir", type=Path, default=None)
    parser.add_argument("--batch_dir", type=Path, default=None)
    parser.add_argument("--sparse_subdir", type=str, default="sparse/0")
    parser.add_argument("--colmap_path", type=str, default=DEFAULT_COLMAP)
    parser.add_argument(
        "--error_type", default="absolute",
        choices=["absolute", "relative"],
    )
    parser.add_argument(
        "--alignment_max_error", type=float, default=ALIGNMENT_MAX_ERROR,
        help="RANSAC inlier threshold for Sim3 alignment (meters)",
    )
    parser.add_argument(
        "--position_accuracy", type=float, default=POSITION_ACCURACY_GT,
        help="GT position accuracy – used as min_error in AUC (meters)",
    )
    parser.add_argument("--ckpt", type=Path, default=None)
    parser.add_argument("--ckpt_subdir", type=str, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if args.batch_dir is not None:
        results = batch_evaluate(
            args.batch_dir, args.sparse_subdir, args.colmap_path,
            args.error_type, args.alignment_max_error, args.position_accuracy,
            ckpt_subdir=args.ckpt_subdir,
        )
        _print_summary(results["summary"], args.error_type)
    elif args.scene_dir is not None:
        sparse_dir = args.sparse_dir or (args.scene_dir / "sparse" / "0")
        result = evaluate_scene(
            args.scene_dir, sparse_dir, args.colmap_path,
            args.error_type, args.alignment_max_error, args.position_accuracy,
            ckpt_path=args.ckpt,
        )
        results = {"per_scene": {args.scene_dir.name: result}}
        print(json.dumps(result, indent=2))
    else:
        parser.error("Specify either --scene_dir or --batch_dir")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


def _print_summary(summary: dict, error_type: str):
    thresholds = ABS_THRESHOLDS if error_type == "absolute" else REL_THRESHOLDS
    unit = "m" if error_type == "absolute" else "deg"
    score = "AUC"

    cols = [f"{score}@{t}{unit}" for t in thresholds]
    header = f"{'Prefix':<15} " + " ".join(f"{c:>12}" for c in cols)
    sep = "=" * len(header)

    print(f"\n{sep}")
    print("ETH3D SLAM Pose Evaluation Summary")
    print(sep)
    print(header)
    print("-" * len(header))

    for prefix in SCENE_PREFIXES:
        if prefix not in summary:
            continue
        m = summary[prefix]
        vals = " ".join(f"{m.get(f'auc_{t}{unit}', 0):>12.2f}" for t in thresholds)
        print(f"{prefix:<15} {vals}")

    print("-" * len(header))
    if "average" in summary:
        m = summary["average"]
        vals = " ".join(f"{m.get(f'auc_{t}{unit}', 0):>12.2f}" for t in thresholds)
        print(f"{'Average':<15} {vals}")
    print(sep)


if __name__ == "__main__":
    main()
