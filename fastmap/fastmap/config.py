from dataclasses import dataclass, field
from loguru import logger
import dacite
import yaml


@dataclass
class DistortionConfig:
    # number of sampling levels
    num_levels: int = 3
    # number of samples per level
    num_samples: int = 10
    # minimum value of the distortion parameter (negative is barrel distortion)
    alpha_min: float = -0.5
    # maximum value of the distortion parameter (positive is pincushion distortion)
    alpha_max: float = 0.2


@dataclass
class FocalConfig:
    # minimum horizontal field of view
    min_fov: float = 20.0
    # maximum horizontal field of view
    max_fov: float = 160.0
    # number of samples for focal estimation
    num_samples: int = 100
    # use to control how spiky the voting should be
    std: float = 0.01


@dataclass
class FundamentalConfig:
    # number of re-weighting iterations for fundamental matrix estimation
    num_iters: int = 10


@dataclass
class HomographyConfig:
    # number of re-weighting iterations for homography matrix estimation
    num_iters: int = 10


@dataclass
class PoseDecompositionConfig:
    # after pose decomposition, point pairs with epipolar error larger than this threshold will be considered as outliers
    error_thr: float = 0.01


@dataclass
class RotationConfig:
    """See the docstring of the `global_rotation()` function in `rotation.py` for more details."""

    max_inlier_thr: int = 128
    min_inlier_thr: int = 16
    min_inlier_increment_frac: float = 0.01
    max_angle_thr: float = 50.0
    min_angle_thr: float = 10.0
    angle_step: float = 10.0
    lr: float = 0.0001
    log_interval: int = 500


@dataclass
class TrackConfig:
    # minimum track size
    min_track_size: int = 2


@dataclass
class RelativeTranslationConfig:
    """See the docstring of the `re_estimate_relative_t` function in `rel_t.py` for more details."""

    num_candidates: int = 512
    epipolar_error_thr: float = 0.01
    ray_angle_thr: float = 2.0
    min_num_inliers: int = 4
    min_degree: int = 2


@dataclass
class GlobalTranslationConfig:
    """See the docstring of the `global_translation()` function in `translation.py` for more details."""

    num_init: int = 3
    log_interval: int = 500


@dataclass
class EpipolarAdjustmentConfig:
    """See the docstring of the `epipolar_adjustment()` function in `epipolar.py` for more details."""

    num_irls_steps: int = 3
    num_prune_steps: int = 3
    max_thr: float = 0.01
    min_thr: float = 0.005
    lr: float = 1e-4
    lr_decay: float = 0.5
    log_interval: int = 500


@dataclass
class SparseReconstructionConfig:
    """See the docstring of the `sparse_reconstruction()` function in `sparse.py` for more details."""

    reproj_err_thr: float = 15.0
    min_ray_angle: float = 2.0
    batch_size: int = 4096 * 16


@dataclass
class Config:
    distortion: DistortionConfig = field(default_factory=DistortionConfig)
    focal: FocalConfig = field(default_factory=FocalConfig)
    fundamental: FundamentalConfig = field(default_factory=FundamentalConfig)
    homography: HomographyConfig = field(default_factory=HomographyConfig)
    pose_decomposition: PoseDecompositionConfig = field(
        default_factory=PoseDecompositionConfig
    )
    rotation: RotationConfig = field(default_factory=RotationConfig)
    track: TrackConfig = field(default_factory=TrackConfig)
    relative_translation: RelativeTranslationConfig = field(
        default_factory=RelativeTranslationConfig
    )
    global_translation: GlobalTranslationConfig = field(
        default_factory=GlobalTranslationConfig
    )
    epipolar_adjustment: EpipolarAdjustmentConfig = field(
        default_factory=EpipolarAdjustmentConfig
    )
    sparse_reconstruction: SparseReconstructionConfig = field(
        default_factory=SparseReconstructionConfig
    )


def load_config(path: str) -> Config:
    """
    Load the YAML configuration file to the Config dataclass.
    """
    cfg = yaml.safe_load(open(path, "r"))
    try:
        cfg = dacite.from_dict(
            data_class=Config, data=cfg, config=dacite.Config(strict=True)
        )
    except dacite.UnexpectedDataError as e:
        logger.error("Invalid option detected in your config")
        logger.error(f"{e}")
        exit(1)
    return cfg
