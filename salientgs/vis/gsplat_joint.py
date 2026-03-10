import sys
import os
current_file_dir = os.path.dirname(__file__)


workspace_dir = os.path.dirname(current_file_dir)


sys.path.append(workspace_dir)

import json
import math
import os
import time
import wandb
from dataclasses import dataclass, field
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Tuple, Union

import imageio
import nerfview
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import tyro
import viser
import yaml
from vis.utils.colmap import Dataset, Parser
from vis.utils.traj import (
    generate_interpolated_path,
    generate_ellipse_path_z,
    generate_spiral_path,
)
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from fused_ssim import fused_ssim
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from typing_extensions import Literal, assert_never
from vis.utils.misc import AppearanceOptModule, CameraOptModule, knn, rgb_to_sh, set_random_seed
from vis.utils.lib_bilagrid import (
    BilateralGrid,
    slice,
    color_correct,
    total_variation_loss,
)

from gsplat.compression import PngCompression
from gsplat.distributed import cli
from gsplat.rendering import rasterization
from gsplat.strategy import DefaultStrategy, MCMCStrategy
from gsplat.strategy.ops import (
    _multinomial_sample,
    _update_param_with_optimizer,
    inject_noise_to_position,
)
from gsplat.relocation import compute_relocation
from gsplat.cuda._wrapper import rasterize_to_indices_in_range
from gsplat.optimizers import SelectiveAdam


def custom_collate_fn(batch: List[Dict]) -> Dict:
    """Custom collate function that handles variable-length tensors by padding."""
    result = {}
    
    # Keys that may have variable lengths and need padding
    variable_keys = {"points", "point_indices", "depths"}
    
    for key in batch[0].keys():
        values = [item[key] for item in batch]
        
        if key in variable_keys:
            # Pad variable-length tensors
            max_len = max(v.shape[0] for v in values)
            padded = []
            for v in values:
                if v.shape[0] < max_len:
                    pad_size = max_len - v.shape[0]
                    if v.dim() == 1:
                        # [M] -> pad to [max_len]
                        v = F.pad(v, (0, pad_size), value=0)
                    else:
                        # [M, D] -> pad to [max_len, D]
                        v = F.pad(v, (0, 0, 0, pad_size), value=0)
                padded.append(v)
            result[key] = torch.stack(padded, dim=0)
            # Also store the original lengths for masking
            result[f"{key}_lengths"] = torch.tensor([item[key].shape[0] for item in batch])
        elif isinstance(values[0], torch.Tensor):
            # Standard stacking for fixed-size tensors
            result[key] = torch.stack(values, dim=0)
        elif isinstance(values[0], (int, float, np.integer, np.floating)):
            # Convert scalars (including numpy scalars) to tensor
            result[key] = torch.tensor(values)
        else:
            # Keep as list for other types (e.g., strings)
            result[key] = values
    
    return result


@dataclass
class ImportanceGuidedMCMCStrategy:
    """Self-contained MCMC strategy with importance-guided proposals.
    
    This strategy follows the paper:
    `3D Gaussian Splatting as Markov Chain Monte Carlo <https://arxiv.org/abs/2404.09591>`_
    
    With extensions for importance-weighted proposal sampling
    and redundancy-guided relocation.
    
    This strategy will:
    - Periodically relocate GSs with low opacity or high redundancy score using
      importance-weighted proposal sampling.
    - Periodically introduce new GSs sampled from an importance-weighted proposal.
    - Periodically perturb the GSs locations.
    """

    # === MCMC base parameters ===
    # Maximum number of GSs
    cap_max: int = 1_000_000
    # MCMC sampling noise learning rate
    noise_lr: float = 5e5
    # Start refining GSs after this fraction of total training (500/30000 ≈ 0.0167)
    refine_start_epoch: float = 0.0167
    # Stop refining GSs after this fraction of total training (25000/30000 ≈ 0.8333)
    refine_stop_epoch: float = 0.8333
    # Stop injecting noise after this fraction of total training (-1.0 = never stop)
    noise_injection_stop_epoch: float = -1.0
    # Refine GSs every this fraction of total training (100/30000 ≈ 0.0033)
    refine_every_epoch: float = 0.0033
    # GSs with opacity below this value will be pruned
    min_opacity: float = 0.005
    # Whether to print verbose information
    verbose: bool = True

    # === Computed iteration values (set by adjust_strategy) ===
    refine_start_iter: int = field(default=0, init=False)
    refine_stop_iter: int = field(default=0, init=False)
    noise_injection_stop_iter: int = field(default=-1, init=False)
    refine_every: int = field(default=0, init=False)

    # === Importance-guided MCMC parameters ===
    # Soft importance threshold for proposal weighting (importance is reported in 0..100 scale)
    importance_thresh: float = 5.0
    # Redundancy score threshold for relocation candidates (use if set)
    redundancy_thresh: Optional[float] = None
    # Legacy name for redundancy threshold
    pruning_thresh: float = 0.9
    # Fallback to standard opacity sampling when no FastGS weights exist
    fallback_to_opacity: bool = True
    # Mix opacity with a uniform prior to avoid under-sampling low-opacity underfit regions.
    # Proposal base weight becomes: (1-mix)*opacity + mix.
    proposal_opacity_mix: float = 0.05

    # === Ablation toggles ===
    # Disable importance weighting for birth (new GS spawning)
    no_birth_weighting: bool = False
    # Disable importance/redundancy weighting for relocation
    no_relocation_weighting: bool = False

    def check_sanity(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
    ):
        """Sanity check for the parameters and optimizers.

        Check if:
            * `params` and `optimizers` have the same keys.
            * Each optimizer has exactly one param_group, corresponding to each parameter.
            * The following keys are present: {"means", "scales", "quats", "opacities"}.

        Raises:
            AssertionError: If any of the above conditions is not met.
        """
        # Base strategy sanity checks
        trainable_params = set(
            [name for name, param in params.items() if param.requires_grad]
        )
        assert trainable_params == set(optimizers.keys()), (
            "trainable parameters and optimizers must have the same keys, "
            f"but got {trainable_params} and {optimizers.keys()}"
        )

        for optimizer in optimizers.values():
            assert len(optimizer.param_groups) == 1, (
                "Each optimizer must have exactly one param_group, "
                "that corresponds to each parameter, "
                f"but got {len(optimizer.param_groups)}"
            )

        # MCMC-specific required keys
        for key in ["means", "scales", "quats", "opacities"]:
            assert key in params, f"{key} is required in params but missing."

    def initialize_state(self) -> Dict[str, Any]:
        """Initialize and return the running state for this strategy."""
        n_max = 51
        binoms = torch.zeros((n_max, n_max))
        for n in range(n_max):
            for k in range(n + 1):
                binoms[n, k] = math.comb(n, k)
        return {
            "binoms": binoms,
            "fastgs_importance": None,
            "fastgs_redundancy": None,
            "fastgs_pruning": None,
        }

    def _compute_importance_weights(
        self, importance: Optional[Tensor]
    ) -> Optional[Tensor]:
        """Compute soft importance weights for proposal sampling."""
        if importance is None:
            return None
        if importance.numel() == 0:
            return None
        # Use a scale-free soft weighting around `importance_thresh` that preserves
        # non-zero mass for exploration while increasing contrast above threshold.
        thresh = float(max(self.importance_thresh, 1e-6))
        x = (importance.float() - thresh) / thresh
        imp = F.softplus(x)
        imp = imp / (imp.mean() + 1e-6)
        return imp

    def step_pre_backward(
        self,
        *args,
        **kwargs,
    ):
        """Callback function to be executed before the `loss.backward()` call."""
        pass

    def step_post_backward(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
        info: Dict[str, Any],
        lr: float,
    ):
        """Post-backward hook with FastGS candidate filtering.
        
        Args:
            lr (float): Learning rate for "means" attribute of the GS.
        """
        state["binoms"] = state["binoms"].to(params["means"].device)
        binoms = state["binoms"]

        if (
            step < self.refine_stop_iter
            and step > self.refine_start_iter
            and step % self.refine_every == 0
        ):
            n_relocated_gs = self._relocate_gs(params, optimizers, binoms, state=state)
            if self.verbose:
                print(f"Step {step}: Relocated {n_relocated_gs} GSs.")

            n_new_gs = self._add_new_gs(params, optimizers, binoms, state=state)
            if self.verbose:
                print(
                    f"Step {step}: Added {n_new_gs} GSs. "
                    f"Now having {len(params['means'])} GSs."
                )

            torch.cuda.empty_cache()

        noise_stop = (
            self.noise_injection_stop_iter
            if self.noise_injection_stop_iter >= 0
            else float("inf")
        )
        if step < noise_stop:
            inject_noise_to_position(
                params=params,
                optimizers=optimizers,
                state={},
                scaler=lr * self.noise_lr,
            )

    @torch.no_grad()
    def _relocate_gs(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        binoms: Tensor,
        state: Optional[Dict[str, Any]] = None,
    ) -> int:
        opacities = torch.sigmoid(params["opacities"])
        dead_mask = opacities.flatten() <= self.min_opacity

        # Also relocate Gaussians with high redundancy score (redundant from multi-view perspective)
        if state is not None and not self.no_relocation_weighting:
            redundancy = state.get("fastgs_redundancy", None)
            if redundancy is None:
                redundancy = state.get("fastgs_pruning", None)
            if redundancy is not None:
                redundancy_thresh = (
                    self.redundancy_thresh
                    if self.redundancy_thresh is not None
                    else self.pruning_thresh
                )
                redundant_mask = redundancy.flatten() > redundancy_thresh
                dead_mask = dead_mask | redundant_mask

        dead_indices = dead_mask.nonzero(as_tuple=True)[0]
        alive_indices = (~dead_mask).nonzero(as_tuple=True)[0]
        n_gs = len(dead_indices)
        if n_gs <= 0 or len(alive_indices) == 0:
            return n_gs

        # Importance-weighted proposal for relocation targets
        weights = opacities[alive_indices].flatten()
        if self.proposal_opacity_mix > 0:
            mix = float(self.proposal_opacity_mix)
            weights = weights * (1.0 - mix) + mix
        if state is not None and not self.no_relocation_weighting:
            importance = state.get("fastgs_importance", None)
            importance_weights = self._compute_importance_weights(importance)
            if importance_weights is not None:
                weights = weights * importance_weights[alive_indices].flatten()
                if weights.sum() <= 0 and self.fallback_to_opacity:
                    weights = opacities[alive_indices].flatten()
                elif weights.sum() <= 0:
                    return n_gs

        sampled_idxs = _multinomial_sample(weights, n_gs, replacement=True)
        sampled_idxs = alive_indices[sampled_idxs]
        new_opacities, new_scales = compute_relocation(
            opacities=opacities[sampled_idxs],
            scales=torch.exp(params["scales"])[sampled_idxs],
            ratios=torch.bincount(sampled_idxs)[sampled_idxs] + 1,
            binoms=binoms,
        )
        eps = torch.finfo(torch.float32).eps
        new_opacities = torch.clamp(new_opacities, max=1.0 - eps, min=self.min_opacity)

        def param_fn(name: str, p: Tensor) -> Tensor:
            if name == "opacities":
                p[sampled_idxs] = torch.logit(new_opacities)
            elif name == "scales":
                p[sampled_idxs] = torch.log(new_scales)
            p[dead_indices] = p[sampled_idxs]
            return torch.nn.Parameter(p, requires_grad=p.requires_grad)

        def optimizer_fn(key: str, v: Tensor) -> Tensor:
            v[sampled_idxs] = 0
            return v

        _update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)
        if state is not None:
            n_gauss = params["means"].shape[0]
            for k, v in state.items():
                if (
                    isinstance(v, torch.Tensor)
                    and v.ndim > 0
                    and v.shape[0] == n_gauss
                ):
                    v[sampled_idxs] = 0
        return n_gs

    @torch.no_grad()
    def _add_new_gs(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        binoms: Tensor,
        state: Optional[Dict[str, Any]] = None,
    ) -> int:
        current_n_points = len(params["means"])
        n_target = min(self.cap_max, int(1.05 * current_n_points))
        n_gs = max(0, n_target - current_n_points)
        if n_gs <= 0:
            return 0

        opacities = torch.sigmoid(params["opacities"])
        weights = opacities.flatten()
        if self.proposal_opacity_mix > 0:
            mix = float(self.proposal_opacity_mix)
            weights = weights * (1.0 - mix) + mix

        if state is not None and not self.no_birth_weighting:
            importance = state.get("fastgs_importance", None)
            importance_weights = self._compute_importance_weights(importance)
            if importance_weights is not None:
                weights = weights * importance_weights.flatten()
                if weights.sum() <= 0:
                    if self.fallback_to_opacity:
                        weights = opacities.flatten()
                    else:
                        return 0

        sampled_idxs = _multinomial_sample(weights, n_gs, replacement=True)
        new_opacities, new_scales = compute_relocation(
            opacities=opacities[sampled_idxs],
            scales=torch.exp(params["scales"])[sampled_idxs],
            ratios=torch.bincount(sampled_idxs)[sampled_idxs] + 1,
            binoms=binoms,
        )
        eps = torch.finfo(torch.float32).eps
        new_opacities = torch.clamp(new_opacities, max=1.0 - eps, min=self.min_opacity)

        def param_fn(name: str, p: Tensor) -> Tensor:
            if name == "opacities":
                p[sampled_idxs] = torch.logit(new_opacities)
            elif name == "scales":
                p[sampled_idxs] = torch.log(new_scales)
            p_new = torch.cat([p, p[sampled_idxs]])
            return torch.nn.Parameter(p_new, requires_grad=p.requires_grad)

        def optimizer_fn(key: str, v: Tensor) -> Tensor:
            v_new = torch.zeros((len(sampled_idxs), *v.shape[1:]), device=v.device)
            return torch.cat([v, v_new])

        _update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)
        return n_gs

@dataclass
class Config:
    # Disable viewer
    disable_viewer: bool = True
    # Close viewer after training
    close_viewer_after_training: bool = False
    # Path to the .pt files. If provide, it will skip training and run evaluation only.
    ckpt: Optional[List[str]] = None
    # Name of compression strategy to use
    compression: Optional[Literal["png"]] = None
    # Render trajectory path
    render_traj_path: str = "interp"

    # Path to the Mip-NeRF 360 dataset
    data_dir: str = "data/360_v2/garden"
    # Name of the image folder
    image_folder_name: str = "images"
    # Downsample factor for the dataset
    data_factor: int = 4
    # Directory to save results
    result_dir: str = "results/garden"
    # Every N images there is a test image (0 = no split, use all for training)
    test_every: int = 8
    # Minimum number of 3D point correspondences for an image to be included (0 = include all registered)
    min_num_points: int = 0
    # Random crop size for training  (experimental)
    patch_size: Optional[int] = None
    # A global scaler that applies to the scene size related parameters
    global_scale: float = 1.0
    # Normalize the world space
    normalize_world_space: bool = True
    # Filter outlier points during normalization (set False to disable)
    filter_outliers: bool = False
    # Camera model
    camera_model: Literal["pinhole", "ortho", "fisheye"] = "pinhole"

    # Port for the viewer server
    port: int = 8080

    # Batch size for training. Learning rates are scaled automatically
    batch_size: int = 1

    # Number of virtual epochs for training
    num_epochs: int = 184
    # A global factor to scale the number of epochs (useful for distributed training)
    epochs_scaler: float = 1.0
    # Epochs to evaluate the model (as fraction of num_epochs, e.g., 0.233 = 23.3%, 1.0 = 100%)
    eval_epochs: List[float] = field(default_factory=lambda: [0.25, 1.0])
    # Epochs to save the model (as fraction of num_epochs)
    save_epochs: List[float] = field(default_factory=lambda: [0.25, 1.0])

    # Initialization strategy
    init_type: str = "sfm"
    # Initial number of GSs. Ignored if using sfm
    init_num_pts: int = 100_000
    # Initial extent of GSs as a multiple of the camera extent. Ignored if using sfm
    init_extent: float = 3.0
    # Degree of spherical harmonics
    sh_degree: int = 3
    # Initial opacity of GS
    init_opa: float = 0.1
    # Initial scale of GS
    init_scale: float = 1.0
    # Weight for SSIM loss
    ssim_lambda: float = 0.2

    # Near plane clipping distance
    near_plane: float = 0.01
    # Far plane clipping distance
    far_plane: float = 1e10

    # Strategy for GS densification
    strategy: Union[DefaultStrategy, MCMCStrategy, ImportanceGuidedMCMCStrategy] = field(
        default_factory=DefaultStrategy
    )
    # Use packed mode for rasterization, this leads to less memory usage but slightly slower.
    packed: bool = False
    # Use sparse gradients for optimization. (experimental)
    sparse_grad: bool = False
    # Use visible adam from Taming 3DGS. (experimental)
    visible_adam: bool = False
    # Anti-aliasing in rasterization. Might slightly hurt quantitative metrics.
    antialiased: bool = False

    # FastGS scoring configuration (importance/redundancy)
    fastgs_num_views: int = 5
    fastgs_loss_thresh: float = 0.1
    # Low-error threshold for redundancy (defaults to 0.5 * fastgs_loss_thresh)
    fastgs_low_loss_thresh: Optional[float] = None
    # Use quantile thresholds by default (more robust than fixed thresholds)
    fastgs_hi_quantile: float = 0.90
    fastgs_lo_quantile: float = 0.05
    # Robust per-view normalization quantiles for L1 error maps
    fastgs_norm_q_low: float = 0.05
    fastgs_norm_q_high: float = 0.95
    # Disable footprint-area normalization in importance/redundancy scoring
    no_footprint_norm: bool = False

    # Use random background for training to discourage transparency
    random_bkgd: bool = False

    # Opacity regularization
    opacity_reg: float = 0.0
    # Scale regularization
    scale_reg: float = 0.0

    # Enable camera optimization.
    pose_opt: bool = True
    # Learning rate for camera optimization
    pose_opt_lr: float = 1e-4
    # Regularization for camera optimization as weight decay
    pose_opt_reg: float = 1e-6
    # Add noise to camera extrinsics. This is only to test the camera pose optimization.
    pose_noise: float = 1e-3

    # Enable appearance optimization. (experimental)
    app_opt: bool = False
    # Appearance embedding dimension
    app_embed_dim: int = 16
    # Learning rate for appearance optimization
    app_opt_lr: float = 1e-3
    # Regularization for appearance optimization as weight decay
    app_opt_reg: float = 1e-6

    # Enable bilateral grid. (experimental)
    use_bilateral_grid: bool = False
    # Shape of the bilateral grid (X, Y, W)
    bilateral_grid_shape: Tuple[int, int, int] = (16, 16, 8)

    # Enable depth loss. (experimental)
    depth_loss: bool = False
    # Weight for depth loss
    depth_lambda: float = 1e-2

    # Enable BA loss.
    ba_loss: bool = True
    # Weight for BA loss
    ba_lambda: float = 1e-4
    # Huber threshold for BA loss
    ba_thres: float = 1.0
    # Use Gaussian means as BA track points (shared parameters)
    ba_tracks_in_splats: bool = False

    # Dump information to tensorboard every this fraction of epoch (0.0033 ≈ 100/30000 steps)
    tb_every_epochs: float = 0.0033
    # Save training images to tensorboard
    tb_save_image: bool = False

    # Enable wandb logging
    use_wandb: bool = True
    # wandb project name
    wandb_project: str = "gsplat-joint"
    # wandb entity name
    wandb_entity: Optional[str] = None
    # wandb run name
    wandb_name: Optional[str] = None

    lpips_net: Literal["vgg", "alex"] = "alex"

    def adjust_epochs(self, factor: float):
        """Scale num_epochs by factor. Other epoch-based params are fractions so they don't need adjustment."""
        self.num_epochs = int(self.num_epochs * factor)

    def get_eval_steps(self, steps_per_epoch: int) -> List[int]:
        """Convert eval_epochs fractions to actual step numbers."""
        total_steps = self.num_epochs * steps_per_epoch
        return [int(frac * total_steps) - 1 for frac in self.eval_epochs]

    def get_save_steps(self, steps_per_epoch: int) -> List[int]:
        """Convert save_epochs fractions to actual step numbers."""
        total_steps = self.num_epochs * steps_per_epoch
        return [int(frac * total_steps) - 1 for frac in self.save_epochs]

    def get_tb_every(self, steps_per_epoch: int) -> int:
        """Convert tb_every_epochs fraction to actual step interval."""
        return max(1, int(self.tb_every_epochs * steps_per_epoch))

    def get_sh_degree_interval(self, steps_per_epoch: int) -> int:
        """SH degree interval as fraction of epoch (default ~3.3% of epoch for 1000/30000)."""
        return max(1, int(0.033 * steps_per_epoch))

    def adjust_strategy(self, steps_per_epoch: int):
        """Adjust strategy parameters based on steps_per_epoch."""
        total_steps = self.num_epochs * steps_per_epoch
        strategy = self.strategy
        if isinstance(strategy, DefaultStrategy):
            # Default: refine_start=500, refine_stop=15000, reset_every=3000, refine_every=100
            # As fractions of 30000: 1.67%, 50%, 10%, 0.33%
            strategy.refine_start_iter = int(0.0167 * total_steps)
            strategy.refine_stop_iter = int(0.5 * total_steps)
            strategy.reset_every = int(0.1 * total_steps)
            strategy.refine_every = int(0.0033 * total_steps)
        elif isinstance(strategy, ImportanceGuidedMCMCStrategy):
            # ImportanceGuidedMCMCStrategy uses epoch-based parameters, convert to iterations
            strategy.refine_start_iter = int(strategy.refine_start_epoch * total_steps)
            strategy.refine_stop_iter = int(strategy.refine_stop_epoch * total_steps)
            strategy.refine_every = max(1, int(strategy.refine_every_epoch * total_steps))
            if strategy.noise_injection_stop_epoch >= 0:
                strategy.noise_injection_stop_iter = int(strategy.noise_injection_stop_epoch * total_steps)
            else:
                strategy.noise_injection_stop_iter = -1
        elif isinstance(strategy, MCMCStrategy):
            # MCMCStrategy uses iteration-based parameters, scale them
            # refine_start_iter: 500/30000 ≈ 0.0167
            # refine_stop_iter: 25000/30000 ≈ 0.8333
            strategy.refine_start_iter = int(0.0167 * total_steps)
            strategy.refine_stop_iter = int(0.8333 * total_steps)
        else:
            raise ValueError(f"Unknown strategy type: {type(strategy)}")


def create_splats_with_optimizers(
    parser: Parser,
    init_type: str = "sfm",
    init_num_pts: int = 100_000,
    init_extent: float = 3.0,
    init_opacity: float = 0.1,
    init_scale: float = 1.0,
    scene_scale: float = 1.0,
    sh_degree: int = 3,
    sparse_grad: bool = False,
    visible_adam: bool = False,
    batch_size: int = 1,
    feature_dim: Optional[int] = None,
    device: str = "cuda",
    world_rank: int = 0,
    world_size: int = 1,
    init_splats: Optional[torch.nn.ParameterDict] = None,
) -> Tuple[torch.nn.ParameterDict, Dict[str, torch.optim.Optimizer]]:
    if init_splats is not None:
        splats = init_splats
    else:
        if init_type == "sfm":
            points = torch.from_numpy(parser.points).float()
            rgbs = torch.from_numpy(parser.points_rgb / 255.0).float()
        elif init_type == "random":
            points = init_extent * scene_scale * (torch.rand((init_num_pts, 3)) * 2 - 1)
            rgbs = torch.rand((init_num_pts, 3))
        else:
            raise ValueError("Please specify a correct init_type: sfm or random")

        # Initialize the GS size to be the average dist of the 3 nearest neighbors
        dist2_avg = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1)  # [N,]
        dist_avg = torch.sqrt(dist2_avg)
        scales = torch.log(dist_avg * init_scale).unsqueeze(-1).repeat(1, 3)  # [N, 3]

        # Distribute the GSs to different ranks (also works for single rank)
        points = points[world_rank::world_size]
        rgbs = rgbs[world_rank::world_size]
        scales = scales[world_rank::world_size]

        N = points.shape[0]
        quats = torch.rand((N, 4))  # [N, 4]
        opacities = torch.logit(torch.full((N,), init_opacity))  # [N,]

        params = [
            # name, value, lr
            ("means", torch.nn.Parameter(points), 1.6e-4 * scene_scale),
            ("scales", torch.nn.Parameter(scales), 5e-3),
            ("quats", torch.nn.Parameter(quats), 1e-3),
            ("opacities", torch.nn.Parameter(opacities), 5e-2),
        ]

        if feature_dim is None:
            # color is SH coefficients.
            colors = torch.zeros((N, (sh_degree + 1) ** 2, 3))  # [N, K, 3]
            colors[:, 0, :] = rgb_to_sh(rgbs)
            params.append(("sh0", torch.nn.Parameter(colors[:, :1, :]), 2.5e-3))
            params.append(("shN", torch.nn.Parameter(colors[:, 1:, :]), 2.5e-3 / 20))
        else:
            # features will be used for appearance and view-dependent shading
            features = torch.rand(N, feature_dim)  # [N, feature_dim]
            params.append(("features", torch.nn.Parameter(features), 2.5e-3))
            colors = torch.logit(rgbs)  # [N, 3]
            params.append(("colors", torch.nn.Parameter(colors), 2.5e-3))

        splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to(device)

    # Scale learning rate based on batch size, reference:
    # https://www.cs.princeton.edu/~smalladi/blog/2024/01/22/SDEs-ScalingRules/
    # Note that this would not make the training exactly equivalent, see
    # https://arxiv.org/pdf/2402.18824v1
    BS = batch_size * world_size
    optimizer_class = None
    if sparse_grad:
        optimizer_class = torch.optim.SparseAdam
    elif visible_adam:
        optimizer_class = SelectiveAdam
    else:
        optimizer_class = torch.optim.Adam

    lr_map = {
        "means": 1.6e-4 * scene_scale,
        "scales": 5e-3,
        "quats": 1e-3,
        "opacities": 5e-2,
        "sh0": 2.5e-3,
        "shN": 2.5e-3 / 20,
        "features": 2.5e-3,
        "colors": 2.5e-3,
    }

    optimizers = {
        name: optimizer_class(
            [{"params": splats[name], "lr": lr_map[name] * math.sqrt(BS), "name": name}],
            eps=1e-15 / math.sqrt(BS),
            # TODO: check betas logic when BS is larger than 10 betas[0] will be zero.
            betas=(1 - BS * (1 - 0.9), 1 - BS * (1 - 0.999)),
        )
        for name in splats.keys()
    }
    return splats, optimizers


class Runner:
    """Engine for training and testing."""

    def __init__(
        self, local_rank: int, world_rank, world_size: int, cfg: Config
    ) -> None:
        set_random_seed(42 + local_rank)

        self.cfg = cfg
        self.world_rank = world_rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = f"cuda:{local_rank}"

        if isinstance(cfg.strategy, ImportanceGuidedMCMCStrategy) and world_size > 1:
            raise ValueError(
                "ImportanceGuidedMCMCStrategy currently supports only single-GPU training."
            )

        # Where to dump results.
        os.makedirs(cfg.result_dir, exist_ok=True)

        # Setup output directories.
        self.ckpt_dir = f"{cfg.result_dir}/ckpts"
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.stats_dir = f"{cfg.result_dir}/stats"
        os.makedirs(self.stats_dir, exist_ok=True)
        self.render_dir = f"{cfg.result_dir}/renders"
        os.makedirs(self.render_dir, exist_ok=True)

        # Tensorboard
        self.writer = SummaryWriter(log_dir=f"{cfg.result_dir}/tb")

        # wandb
        if cfg.use_wandb and world_rank == 0:
            wandb.init(
                project=cfg.wandb_project,
                entity=cfg.wandb_entity,
                name=cfg.wandb_name,
                config=vars(cfg),
            )

        # Load data: Training data should contain initial points and colors.
        self.parser = Parser(
            data_dir=cfg.data_dir,
            factor=cfg.data_factor,
            normalize=cfg.normalize_world_space,
            test_every=cfg.test_every,
            image_folder_name=cfg.image_folder_name,
            min_num_points=cfg.min_num_points,
            filter_outliers=cfg.filter_outliers,
        )
        self.trainset = Dataset(
            self.parser,
            split="train",
            patch_size=cfg.patch_size,
            load_depths=cfg.depth_loss or cfg.ba_loss,
        )
        self.valset = Dataset(self.parser, split="val", load_depths=cfg.depth_loss or cfg.ba_loss)
        self.scene_scale = self.parser.scene_scale * 1.1 * cfg.global_scale
        print("Scene scale:", self.scene_scale)

        # Model
        feature_dim = 32 if cfg.app_opt else None

        # Load from previous checkpoint if available
        ckpt = None
        if cfg.ckpt is None:
            import glob

            ckpt_files = glob.glob(
                os.path.join(self.ckpt_dir, f"ckpt_*_rank{self.world_rank}.pt")
            )
            if len(ckpt_files) > 0:
                ckpt_files.sort(key=lambda x: int(os.path.basename(x).split("_")[1]))
                latest_ckpt = ckpt_files[-1]
                print(f"Loading latest checkpoint: {latest_ckpt}")
                ckpt = torch.load(latest_ckpt, map_location=self.device)

        init_splats = None
        if ckpt is not None:
            init_splats = torch.nn.ParameterDict().to(self.device)
            for k, v in ckpt["splats"].items():
                init_splats[k] = torch.nn.Parameter(v)

        self.splats, self.optimizers = create_splats_with_optimizers(
            self.parser,
            init_type=cfg.init_type,
            init_num_pts=cfg.init_num_pts,
            init_extent=cfg.init_extent,
            init_opacity=cfg.init_opa,
            init_scale=cfg.init_scale,
            scene_scale=self.scene_scale,
            sh_degree=cfg.sh_degree,
            sparse_grad=cfg.sparse_grad,
            visible_adam=cfg.visible_adam,
            batch_size=cfg.batch_size,
            feature_dim=feature_dim,
            device=self.device,
            world_rank=world_rank,
            world_size=world_size,
            init_splats=init_splats,
        )
        print("Model initialized. Number of GS:", len(self.splats["means"]))

        if ckpt is not None:
            self.start_step = ckpt["step"] + 1
        else:
            self.start_step = 0

        # Densification Strategy
        self.cfg.strategy.check_sanity(self.splats, self.optimizers)

        if isinstance(self.cfg.strategy, DefaultStrategy):
            self.strategy_state = self.cfg.strategy.initialize_state(
                scene_scale=self.scene_scale
            )
        elif isinstance(self.cfg.strategy, (MCMCStrategy, ImportanceGuidedMCMCStrategy)):
            self.strategy_state = self.cfg.strategy.initialize_state()
        else:
            raise ValueError(f"Unknown strategy type: {type(self.cfg.strategy)}")

        # Compression Strategy
        self.compression_method = None
        if cfg.compression is not None:
            if cfg.compression == "png":
                self.compression_method = PngCompression()
            else:
                raise ValueError(f"Unknown compression strategy: {cfg.compression}")

        self.pose_optimizers = []
        self.pose_optimizers_test = []
        if cfg.pose_opt:
            # Pose adjustment for training images
            self.pose_adjust = CameraOptModule(len(self.trainset)).to(self.device)
            self.pose_adjust.zero_init()
            if ckpt is not None and "pose_adjust" in ckpt:
                print("Loading pose_adjust from checkpoint")
                self.pose_adjust.load_state_dict(ckpt["pose_adjust"])
            self.pose_optimizers = [
                torch.optim.Adam(
                    self.pose_adjust.parameters(),
                    lr=cfg.pose_opt_lr * math.sqrt(cfg.batch_size),
                    weight_decay=cfg.pose_opt_reg,
                )
            ]
            if world_size > 1:
                self.pose_adjust = DDP(self.pose_adjust)
            
            # Pose adjustment for test images (separate module)
            self.pose_adjust_test = CameraOptModule(len(self.valset)).to(self.device)
            self.pose_adjust_test.zero_init()
            if ckpt is not None and "pose_adjust_test" in ckpt:
                print("Loading pose_adjust_test from checkpoint")
                self.pose_adjust_test.load_state_dict(ckpt["pose_adjust_test"])
            self.pose_optimizers_test = [
                torch.optim.Adam(
                    self.pose_adjust_test.parameters(),
                    lr=cfg.pose_opt_lr * math.sqrt(cfg.batch_size),
                    weight_decay=cfg.pose_opt_reg,
                )
            ]
            if world_size > 1:
                self.pose_adjust_test = DDP(self.pose_adjust_test)

        if cfg.pose_noise > 0.0:
            self.pose_perturb = CameraOptModule(len(self.trainset)).to(self.device)
            self.pose_perturb.random_init(cfg.pose_noise)
            if world_size > 1:
                self.pose_perturb = DDP(self.pose_perturb)

        self.app_optimizers = []
        if cfg.app_opt:
            assert feature_dim is not None
            self.app_module = AppearanceOptModule(
                len(self.trainset), feature_dim, cfg.app_embed_dim, cfg.sh_degree
            ).to(self.device)
            # initialize the last layer to be zero so that the initial output is zero.
            torch.nn.init.zeros_(self.app_module.color_head[-1].weight)
            torch.nn.init.zeros_(self.app_module.color_head[-1].bias)
            if ckpt is not None and "app_module" in ckpt:
                print("Loading app_module from checkpoint")
                self.app_module.load_state_dict(ckpt["app_module"])
            self.app_optimizers = [
                torch.optim.Adam(
                    self.app_module.embeds.parameters(),
                    lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size) * 10.0,
                    weight_decay=cfg.app_opt_reg,
                ),
                torch.optim.Adam(
                    self.app_module.color_head.parameters(),
                    lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size),
                ),
            ]
            if world_size > 1:
                self.app_module = DDP(self.app_module)

        self.bil_grid_optimizers = []
        if cfg.use_bilateral_grid:
            self.bil_grids = BilateralGrid(
                len(self.trainset),
                grid_X=cfg.bilateral_grid_shape[0],
                grid_Y=cfg.bilateral_grid_shape[1],
                grid_W=cfg.bilateral_grid_shape[2],
            ).to(self.device)
            self.bil_grid_optimizers = [
                torch.optim.Adam(
                    self.bil_grids.parameters(),
                    lr=2e-3 * math.sqrt(cfg.batch_size),
                    eps=1e-15,
                ),
            ]

        # Losses & Metrics.
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)

        if cfg.lpips_net == "alex":
            self.lpips = LearnedPerceptualImagePatchSimilarity(
                net_type="alex", normalize=True
            ).to(self.device)
        elif cfg.lpips_net == "vgg":
            # The 3DGS official repo uses lpips vgg, which is equivalent with the following:
            self.lpips = LearnedPerceptualImagePatchSimilarity(
                net_type="vgg", normalize=False
            ).to(self.device)
        else:
            raise ValueError(f"Unknown LPIPS network: {cfg.lpips_net}")

        # BA track points - separate from splats by default
        self.track_points_3d = None
        self.ba_optimizers = []
        if cfg.ba_loss:
            if cfg.ba_tracks_in_splats:
                # Reuse Gaussian means as track points (shared parameters)
                self.track_points_3d = self.splats["means"]
            else:
                # Initialize separate track points from SfM points
                track_points = torch.from_numpy(self.parser.points).float().to(self.device)
                self.track_points_3d = torch.nn.Parameter(track_points)
                self.ba_optimizers = [
                    torch.optim.Adam(
                        [self.track_points_3d],
                        lr=1e-4 * self.scene_scale * math.sqrt(cfg.batch_size),
                        eps=1e-15,
                    )
                ]

        # Viewer
        if not self.cfg.disable_viewer:
            self.server = viser.ViserServer(port=cfg.port, verbose=False)
            self.viewer = nerfview.Viewer(
                server=self.server,
                render_fn=self._viewer_render_fn,
                mode="training",
            )

    def rasterize_splats(
        self,
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        masks: Optional[Tensor] = None,
        freeze_gs: bool = False,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Dict]:
        # Optionally freeze GS parameters (for test pose optimization)
        if freeze_gs:
            means = self.splats["means"].detach()  # [N, 3]
            quats = self.splats["quats"].detach()  # [N, 4]
            scales = torch.exp(self.splats["scales"].detach())  # [N, 3]
            opacities = torch.sigmoid(self.splats["opacities"].detach())  # [N,]
        else:
            means = self.splats["means"]  # [N, 3]
            # quats = F.normalize(self.splats["quats"], dim=-1)  # [N, 4]
            # rasterization does normalization internally
            quats = self.splats["quats"]  # [N, 4]
            scales = torch.exp(self.splats["scales"])  # [N, 3]
            opacities = torch.sigmoid(self.splats["opacities"])  # [N,]

        image_ids = kwargs.pop("image_ids", None)
        if self.cfg.app_opt:
            features = self.splats["features"].detach() if freeze_gs else self.splats["features"]
            splat_colors = self.splats["colors"].detach() if freeze_gs else self.splats["colors"]
            colors = self.app_module(
                features=features,
                embed_ids=image_ids,
                dirs=means[None, :, :] - camtoworlds[:, None, :3, 3],
                sh_degree=kwargs.pop("sh_degree", self.cfg.sh_degree),
            )
            colors = colors + splat_colors
            colors = torch.sigmoid(colors)
        else:
            sh0 = self.splats["sh0"].detach() if freeze_gs else self.splats["sh0"]
            shN = self.splats["shN"].detach() if freeze_gs else self.splats["shN"]
            colors = torch.cat([sh0, shN], 1)  # [N, K, 3]

        rasterize_mode = "antialiased" if self.cfg.antialiased else "classic"
        packed = kwargs.pop("packed", self.cfg.packed)
        render_colors, render_alphas, info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=torch.linalg.inv(camtoworlds),  # [C, 4, 4]
            Ks=Ks,  # [C, 3, 3]
            width=width,
            height=height,
            packed=packed,
            absgrad=(
                self.cfg.strategy.absgrad
                if isinstance(self.cfg.strategy, DefaultStrategy)
                else False
            ),
            sparse_grad=self.cfg.sparse_grad,
            rasterize_mode=rasterize_mode,
            distributed=self.world_size > 1,
            camera_model=self.cfg.camera_model,
            **kwargs,
        )
        if masks is not None:
            render_colors[~masks] = 0
        return render_colors, render_alphas, info

    @torch.no_grad()
    def compute_fastgs_scores(
        self,
        sh_degree: Optional[int] = None,
        num_views: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Compute FastGS importance/redundancy scores using multi-view error maps.

        Notes:
            - Uses robust per-view normalization (quantiles) rather than min-max.
            - Uses quantile-based high/low masks to keep selection selective across training.
            - Normalizes per-Gaussian attribution by an estimate of footprint size to reduce
              bias toward large projected Gaussians.
        """
        cfg = self.cfg
        device = self.device
        total_views = len(self.trainset)
        if total_views == 0:
            n_gauss = len(self.splats["means"])
            zeros = torch.zeros(n_gauss, device=device)
            return zeros, zeros

        if num_views is None:
            num_views = cfg.fastgs_num_views
        num_views = min(num_views, total_views)
        if num_views <= 0:
            n_gauss = len(self.splats["means"])
            zeros = torch.zeros(n_gauss, device=device)
            return zeros, zeros

        if sh_degree is None:
            sh_degree = cfg.sh_degree

        replace = num_views > total_views
        view_indices = np.random.choice(total_views, size=num_views, replace=replace)

        n_gauss = len(self.splats["means"])
        imp_accum = torch.zeros(n_gauss, device=device, dtype=torch.float32)
        red_accum = torch.zeros(n_gauss, device=device, dtype=torch.float32)
        area_accum = torch.zeros(n_gauss, device=device, dtype=torch.float32)

        for idx in view_indices:
            data = self.trainset[int(idx)]
            camtoworlds = data["camtoworld"].to(device).unsqueeze(0)
            Ks = data["K"].to(device).unsqueeze(0)
            pixels = data["image"].to(device) / 255.0
            height, width = pixels.shape[0], pixels.shape[1]

            masks = data.get("mask", None)
            if masks is not None:
                masks = masks.to(device).unsqueeze(0)

            render_colors, render_alphas, info = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                masks=masks,
                sh_degree=sh_degree,
                packed=False,
            )

            colors = render_colors[0]
            alphas = render_alphas[0]
            if cfg.random_bkgd:
                bkgd = torch.rand(1, 3, device=device)
                colors = colors + bkgd * (1.0 - alphas)

            gt = pixels
            if masks is not None:
                mask_img = masks[0]
                colors = colors * mask_img[..., None]
                gt = gt * mask_img[..., None]
            else:
                mask_img = None

            l1_map = torch.mean(torch.abs(colors - gt), dim=-1)
            if mask_img is not None:
                valid_mask = mask_img.bool()
            else:
                valid_mask = torch.ones_like(l1_map, dtype=torch.bool, device=device)

            valid_vals = l1_map[valid_mask]
            if valid_vals.numel() == 0:
                continue

            # Robust per-view normalization via quantiles (avoid min-max artifacts)
            q_low = float(cfg.fastgs_norm_q_low)
            q_high = float(cfg.fastgs_norm_q_high)
            q_low = max(0.0, min(1.0, q_low))
            q_high = max(0.0, min(1.0, q_high))
            if q_high <= q_low:
                q_low, q_high = 0.05, 0.95

            low_val = torch.quantile(valid_vals, q_low)
            high_val = torch.quantile(valid_vals, q_high)
            denom = (high_val - low_val).clamp_min(1e-6)
            l1_norm = ((l1_map - low_val) / denom).clamp(0.0, 1.0)
            l1_norm = l1_norm * valid_mask.float()

            # Selective high/low masks via quantiles (default) with fallback to fixed thresholds
            if cfg.fastgs_hi_quantile is not None and cfg.fastgs_lo_quantile is not None:
                q_hi = max(0.0, min(1.0, float(cfg.fastgs_hi_quantile)))
                q_lo = max(0.0, min(1.0, float(cfg.fastgs_lo_quantile)))
                if q_hi <= q_lo:
                    q_lo = max(0.0, min(1.0, q_hi - 0.1))
                hi_th = torch.quantile(l1_norm[valid_mask], q_hi)
                lo_th = torch.quantile(l1_norm[valid_mask], q_lo)
            else:
                hi_th = torch.tensor(float(cfg.fastgs_loss_thresh), device=device)
                low_thresh = cfg.fastgs_low_loss_thresh
                if low_thresh is None:
                    low_thresh = float(cfg.fastgs_loss_thresh) * 0.5
                lo_th = torch.tensor(float(low_thresh), device=device)

            lo_th = torch.minimum(lo_th, hi_th - 1e-4)
            lo_th = torch.clamp(lo_th, 0.0, 1.0)
            hi_th = torch.clamp(hi_th, 0.0, 1.0)

            # Per-pixel normalized weights (0..1), then aggregate per-Gaussian and normalize by footprint area
            hi_weight = (l1_norm - hi_th).clamp_min(0.0) / (1.0 - hi_th + 1e-6)
            lo_weight = (lo_th - l1_norm).clamp_min(0.0) / (lo_th + 1e-6)

            # Approximate footprint size using projected radii (reduces bias toward large footprints)
            if not cfg.no_footprint_norm:
                radii = info.get("radii", None)
                if radii is not None:
                    radii_v = radii[0].to(torch.float32)  # [N, 2]
                    footprint = radii_v[:, 0].clamp_min(0) * radii_v[:, 1].clamp_min(0)
                    area_accum += footprint
            else:
                area_accum += torch.ones(n_gauss, device=device)

            hi_mask = hi_weight > 0
            lo_mask = lo_weight > 0
            hi_weight_flat = hi_weight.reshape(-1)
            lo_weight_flat = lo_weight.reshape(-1)

            def _accum_from_mask(mask: Tensor, w_flat: Tensor, out: Tensor) -> None:
                if not mask.any():
                    return
                transmittances = mask.float().unsqueeze(0)
                gaussian_ids, pixel_ids, _ = rasterize_to_indices_in_range(
                    0,
                    1_000_000_000,
                    transmittances,
                    info["means2d"],
                    info["conics"],
                    info["opacities"],
                    width,
                    height,
                    info["tile_size"],
                    info["isect_offsets"],
                    info["flatten_ids"],
                )
                if gaussian_ids.numel() == 0:
                    return
                gaussian_ids = gaussian_ids.to(dtype=torch.int64)
                pixel_ids = pixel_ids.to(dtype=torch.int64)
                weights = w_flat[pixel_ids].to(dtype=torch.float32)
                out.scatter_add_(0, gaussian_ids, weights)

            _accum_from_mask(hi_mask & valid_mask, hi_weight_flat, imp_accum)
            _accum_from_mask(lo_mask & valid_mask, lo_weight_flat, red_accum)

        if area_accum.sum() <= 0:
            zeros = torch.zeros(n_gauss, device=device)
            return zeros, zeros

        importance_score = torch.clamp(imp_accum / (area_accum + 1e-6), 0.0, 1.0) * 100.0
        redundancy_raw = torch.clamp(red_accum / (area_accum + 1e-6), 0.0, 1.0)
        # Normalize redundancy to [0,1] for stable thresholding across scenes
        min_red = redundancy_raw.min()
        max_red = redundancy_raw.max()
        red_denom = (max_red - min_red).clamp_min(1e-6)
        redundancy_score = (redundancy_raw - min_red) / red_denom
        return importance_score, redundancy_score

    def compute_ba_loss(
        self,
        data: Dict,
        camtoworlds: torch.Tensor,
        Ks: torch.Tensor,
        freeze_points: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Compute bundle adjustment loss for reprojection consistency.
        
        Args:
            data: Dict containing 'point_indices', 'points', and optionally 'points_lengths'
            camtoworlds: Camera-to-world transforms [B, 4, 4]
            Ks: Intrinsic matrices [B, 3, 3]
            freeze_points: If True, 3D points are frozen (for test pose optimization)
        
        Returns:
            baloss: Scalar BA loss
            reproj_errors: Per-observation reprojection errors (for logging), or None
            per_obs_loss: Per-observation loss values (for logging), or None
            point_indices: Valid point indices (for logging), or None
        """
        device = self.device
        cfg = self.cfg
        
        point_indices = data["point_indices"].to(device)  # [B, M]
        points_2d_obs = data["points"].to(device)  # [B, M, 2]
        
        # Get valid mask for padded data (when batch_size > 1)
        if "points_lengths" in data:
            lengths = data["points_lengths"].to(device)  # [B]
            max_len = points_2d_obs.shape[1]
            valid_mask = torch.arange(max_len, device=device)[None, :] < lengths[:, None]
        else:
            valid_mask = torch.ones(point_indices.shape, dtype=torch.bool, device=device)
        
        # Get 3D track points
        if self.track_points_3d is None:
            raise ValueError("BA loss requested but track points are not initialized.")
        
        safe_indices = point_indices
        if cfg.ba_tracks_in_splats:
            max_idx = self.track_points_3d.shape[0]
            if max_idx == 0:
                raise ValueError("No Gaussian means available for BA track points.")
            in_range_mask = (point_indices >= 0) & (point_indices < max_idx)
            valid_mask = valid_mask & in_range_mask
            safe_indices = point_indices.clamp(0, max_idx - 1)

        if freeze_points:
            with torch.no_grad():
                points_3d = self.track_points_3d[safe_indices]  # [B, M, 3]
        else:
            points_3d = self.track_points_3d[safe_indices]  # [B, M, 3]
        
        # Transform to camera space
        worldtocams = torch.inverse(camtoworlds)
        points_cam = torch.matmul(worldtocams[:, :3, :3], points_3d.transpose(1, 2)) + worldtocams[:, :3, 3:4]
        points_cam = points_cam.transpose(1, 2)  # [B, M, 3]
        
        # Pinhole projection
        points_proj = torch.matmul(Ks, points_cam.transpose(1, 2)).transpose(1, 2)
        points_2d_proj = points_proj[..., :2] / (points_proj[..., 2:3] + 1e-10)
        
        # Compute reprojection residuals with Huber robust cost
        residuals = points_2d_proj - points_2d_obs  # [B, M, 2]
        
        # Per-observation reprojection error (L2 norm of residual)
        reproj_errors = torch.norm(residuals, dim=-1)  # [B, M]
        
        # Per-observation Huber loss (no reduction)
        per_obs_huber = F.huber_loss(residuals, torch.zeros_like(residuals), 
                                     delta=cfg.ba_thres, reduction='none')  # [B, M, 2]
        per_obs_loss = per_obs_huber.sum(dim=-1)  # [B, M] - sum over x,y
        
        # Mask out padded values and compute mean only over valid observations
        per_obs_loss_masked = per_obs_loss * valid_mask.float()
        num_valid = valid_mask.sum()
        
        baloss = per_obs_loss_masked.sum() / (num_valid + 1e-10)
        
        # Return logging info (only valid observations)
        return (
            baloss,
            reproj_errors[valid_mask].detach(),
            per_obs_loss_masked[valid_mask].detach(),
            point_indices[valid_mask].detach(),
        )

    def train(self):
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank
        world_size = self.world_size

        # Dump cfg.
        if world_rank == 0:
            with open(f"{cfg.result_dir}/cfg.yml", "w") as f:
                yaml.dump(vars(cfg), f)

        init_step = self.start_step

        # Use custom collate function if we have variable-length data (depth_loss or ba_loss)
        collate_fn = custom_collate_fn if (cfg.depth_loss or cfg.ba_loss) else None
        trainloader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=8,
            persistent_workers=True,
            pin_memory=True,
            collate_fn=collate_fn,
        )
        
        # Compute steps per epoch and total steps from virtual epochs
        steps_per_epoch = len(trainloader)
        effective_max_steps = cfg.num_epochs * steps_per_epoch
        
        # # Ensure minimum of 30000 steps
        # if effective_max_steps < 30000:
        #     effective_max_steps = 30000
        
        # Compute step-based parameters from epoch fractions
        eval_steps = cfg.get_eval_steps(steps_per_epoch)
        save_steps = cfg.get_save_steps(steps_per_epoch)
        tb_every = cfg.get_tb_every(steps_per_epoch)
        sh_degree_interval = cfg.get_sh_degree_interval(steps_per_epoch)
        
        # Adjust strategy parameters based on total steps
        cfg.adjust_strategy(steps_per_epoch)
        
        # Warmup steps for bilateral grid (3.3% of total steps)
        warmup_steps = int(0.033 * effective_max_steps)
        
        print(f"Virtual epochs: {cfg.num_epochs}, steps per epoch: {steps_per_epoch}, "
              f"total steps: {effective_max_steps}")
        print(f"Eval at steps: {eval_steps}, Save at steps: {save_steps}, "
              f"TB every: {tb_every} steps, SH interval: {sh_degree_interval} steps")

        # Running average tracker for losses (window = steps_per_epoch)
        running_avg = {
            "loss": deque(maxlen=steps_per_epoch),
            "l1loss": deque(maxlen=steps_per_epoch),
            "ssimloss": deque(maxlen=steps_per_epoch),
        }
        if cfg.depth_loss:
            running_avg["depthloss"] = deque(maxlen=steps_per_epoch)
        if cfg.ba_loss:
            running_avg["baloss"] = deque(maxlen=steps_per_epoch)

        schedulers = [
            # means has a learning rate schedule, that end at 0.01 of the initial value
            torch.optim.lr_scheduler.ExponentialLR(
                self.optimizers["means"], gamma=0.01 ** (1.0 / effective_max_steps)
            ),
        ]
        if cfg.pose_opt:
            # pose optimization has a learning rate schedule
            schedulers.append(
                torch.optim.lr_scheduler.ExponentialLR(
                    self.pose_optimizers[0], gamma=0.01 ** (1.0 / effective_max_steps)
                )
            )
            # test pose optimization has same schedule
            schedulers.append(
                torch.optim.lr_scheduler.ExponentialLR(
                    self.pose_optimizers_test[0], gamma=0.01 ** (1.0 / effective_max_steps)
                )
            )
        if cfg.use_bilateral_grid:
            # bilateral grid has a learning rate schedule. Linear warmup then decay.
            schedulers.append(
                torch.optim.lr_scheduler.ChainedScheduler(
                    [
                        torch.optim.lr_scheduler.LinearLR(
                            self.bil_grid_optimizers[0],
                            start_factor=0.01,
                            total_iters=warmup_steps,
                        ),
                        torch.optim.lr_scheduler.ExponentialLR(
                            self.bil_grid_optimizers[0], gamma=0.01 ** (1.0 / effective_max_steps)
                        ),
                    ]
                )
            )
        if cfg.ba_loss and self.ba_optimizers:
            # BA optimizer has a learning rate schedule, decays to 1% of initial value
            schedulers.append(
                torch.optim.lr_scheduler.ExponentialLR(
                    self.ba_optimizers[0], gamma=0.01 ** (1.0 / effective_max_steps)
                )
            )

        # Test dataloader for test pose optimization (freeze GS, only optimize pose)
        if cfg.pose_opt:
            testloader = torch.utils.data.DataLoader(
                self.valset,
                batch_size=cfg.batch_size,
                shuffle=True,
                num_workers=8,
                persistent_workers=True,
                pin_memory=True,
            )

        # Training loop with virtual epochs
        global_tic = time.time()
        step = init_step
        pbar = tqdm.tqdm(total=effective_max_steps, initial=init_step)
        for epoch in range(cfg.num_epochs):
            if step >= effective_max_steps:
                break
            for data in trainloader:
                if step >= effective_max_steps:
                    break
                if not cfg.disable_viewer:
                    while self.viewer.state.status == "paused":
                        time.sleep(0.01)
                    self.viewer.lock.acquire()
                    tic = time.time()

                camtoworlds = camtoworlds_gt = data["camtoworld"].to(device)  # [1, 4, 4]
                Ks = data["K"].to(device)  # [1, 3, 3]
                pixels = data["image"].to(device) / 255.0  # [1, H, W, 3]
                num_train_rays_per_step = (
                    pixels.shape[0] * pixels.shape[1] * pixels.shape[2]
                )
                image_ids = data["image_id"].to(device)
                abs_indices = data["index"].to(device)
                masks = data["mask"].to(device) if "mask" in data else None  # [1, H, W]
                if cfg.depth_loss:
                    points = data["points"].to(device)  # [1, M, 2]
                    depths_gt = data["depths"].to(device)  # [1, M]

                height, width = pixels.shape[1:3]

                if cfg.pose_noise:
                    # Anneal pose noise: scale decreases from 1.0 to 0.01 over training
                    noise_scale = 0.01 ** (step / effective_max_steps)
                    camtoworlds_perturbed = self.pose_perturb(camtoworlds, image_ids)
                    # Interpolate between original and perturbed poses based on noise_scale
                    camtoworlds = camtoworlds + noise_scale * (camtoworlds_perturbed - camtoworlds)

                if cfg.pose_opt:
                    camtoworlds = self.pose_adjust(camtoworlds, image_ids)

                # Fix the pose of camera 0
                mask = (abs_indices == 0)
                if mask.any():
                    camtoworlds = torch.where(mask[..., None, None], camtoworlds_gt, camtoworlds)

                # sh schedule
                sh_degree_to_use = min(step // sh_degree_interval, cfg.sh_degree)

                # forward
                renders, alphas, info = self.rasterize_splats(
                    camtoworlds=camtoworlds,
                    Ks=Ks,
                    width=width,
                    height=height,
                    sh_degree=sh_degree_to_use,
                    near_plane=cfg.near_plane,
                    far_plane=cfg.far_plane,
                    image_ids=image_ids,
                    render_mode="RGB+ED" if cfg.depth_loss else "RGB",
                    masks=masks,
                )
                if renders.shape[-1] == 4:
                    colors, depths = renders[..., 0:3], renders[..., 3:4]
                else:
                    colors, depths = renders, None

                if cfg.use_bilateral_grid:
                    grid_y, grid_x = torch.meshgrid(
                        (torch.arange(height, device=self.device) + 0.5) / height,
                        (torch.arange(width, device=self.device) + 0.5) / width,
                        indexing="ij",
                    )
                    grid_xy = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
                    colors = slice(self.bil_grids, grid_xy, colors, image_ids)["rgb"]

                if cfg.random_bkgd:
                    bkgd = torch.rand(1, 3, device=device)
                    colors = colors + bkgd * (1.0 - alphas)

                self.cfg.strategy.step_pre_backward(
                    params=self.splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=info,
                )

                # loss
                l1loss = F.l1_loss(colors, pixels)
                ssimloss = 1.0 - fused_ssim(
                    colors.permute(0, 3, 1, 2), pixels.permute(0, 3, 1, 2), padding="valid"
                )
                loss = l1loss * (1.0 - cfg.ssim_lambda) + ssimloss * cfg.ssim_lambda
                if cfg.depth_loss:
                    # query depths from depth map
                    points_normalized = torch.stack(
                        [
                            points[:, :, 0] / (width - 1) * 2 - 1,
                            points[:, :, 1] / (height - 1) * 2 - 1,
                        ],
                        dim=-1,
                    )  # normalize to [-1, 1]
                    grid = points_normalized.unsqueeze(2)  # [B, M, 1, 2]
                    sampled_depths = F.grid_sample(
                        depths.permute(0, 3, 1, 2), grid, align_corners=True
                    )  # [B, 1, M, 1]
                    sampled_depths = sampled_depths.squeeze(3).squeeze(1)  # [B, M]
                    
                    # Get valid mask for padded data (when batch_size > 1)
                    if "points_lengths" in data:
                        lengths = data["points_lengths"].to(device)  # [B]
                        max_len = depths_gt.shape[1]
                        depth_valid_mask = torch.arange(max_len, device=device)[None, :] < lengths[:, None]
                    else:
                        depth_valid_mask = torch.ones_like(depths_gt, dtype=torch.bool)
                    
                    # calculate loss in disparity space
                    disp = torch.where(sampled_depths > 0.0, 1.0 / sampled_depths, torch.zeros_like(sampled_depths))
                    disp_gt = 1.0 / depths_gt  # [B, M]
                    
                    # Apply mask and compute mean only over valid observations
                    disp_diff = torch.abs(disp - disp_gt) * depth_valid_mask.float()
                    num_valid_depth = depth_valid_mask.sum()
                    depthloss = (disp_diff.sum() / (num_valid_depth + 1e-10)) * self.scene_scale
                    loss += depthloss * cfg.depth_lambda
                if cfg.use_bilateral_grid:
                    tvloss = 10 * total_variation_loss(self.bil_grids.grids)
                    loss += tvloss

                # regularizations
                if cfg.opacity_reg > 0.0:
                    loss = (
                        loss
                        + cfg.opacity_reg
                        * torch.abs(torch.sigmoid(self.splats["opacities"])).mean()
                    )
                if cfg.scale_reg > 0.0:
                    loss = (
                        loss
                        + cfg.scale_reg * torch.abs(torch.exp(self.splats["scales"])).mean()
                    )

                if cfg.ba_loss:
                    # BA loss: optimize 3D track points for reprojection consistency
                    baloss, reproj_errors, per_obs_loss, point_indices_valid = self.compute_ba_loss(
                        data, camtoworlds, Ks, freeze_points=False
                    )
                    # Store for gradient analysis
                    self._ba_reproj_errors = reproj_errors
                    self._ba_per_obs_loss = per_obs_loss
                    self._ba_point_indices = point_indices_valid
                    loss += baloss * cfg.ba_lambda

                # Update running averages
                running_avg["loss"].append(loss.item())
                running_avg["l1loss"].append(l1loss.item())
                running_avg["ssimloss"].append(ssimloss.item())
                if cfg.depth_loss:
                    running_avg["depthloss"].append(depthloss.item())
                if cfg.ba_loss:
                    running_avg["baloss"].append(baloss.item())

                loss.backward()

                # Compute running averages for display
                avg_loss = sum(running_avg["loss"]) / len(running_avg["loss"]) if running_avg["loss"] else 0
                avg_l1 = sum(running_avg["l1loss"]) / len(running_avg["l1loss"]) if running_avg["l1loss"] else 0
                avg_ssim = sum(running_avg["ssimloss"]) / len(running_avg["ssimloss"]) if running_avg["ssimloss"] else 0
                
                desc = f"epoch={epoch}| loss={loss.item():.3f} (avg={avg_loss:.3f})| " f"sh degree={sh_degree_to_use}| "
                if cfg.depth_loss:
                    avg_depth = sum(running_avg["depthloss"]) / len(running_avg["depthloss"]) if running_avg["depthloss"] else 0
                    desc += f"depth={depthloss.item():.6f} (avg={avg_depth:.6f})| "
                if cfg.ba_loss:
                    avg_ba = sum(running_avg["baloss"]) / len(running_avg["baloss"]) if running_avg["baloss"] else 0
                    desc += f"ba={baloss.item():.6f} (avg={avg_ba:.6f})| "
                if cfg.pose_opt:
                    with torch.no_grad():
                        if self.world_size > 1:
                            embeds = self.pose_adjust.module.embeds.weight
                        else:
                            embeds = self.pose_adjust.embeds.weight
                        dist = torch.norm(embeds[:, :3], dim=-1).mean()
                        desc += f"pose dist={dist.item():.6f}| "
                pbar.set_description(desc)

                # write images (gt and render)
                # if world_rank == 0 and step % 800 == 0:
                #     canvas = torch.cat([pixels, colors], dim=2).detach().cpu().numpy()
                #     canvas = canvas.reshape(-1, *canvas.shape[2:])
                #     imageio.imwrite(
                #         f"{self.render_dir}/train_rank{self.world_rank}.png",
                #         (canvas * 255).astype(np.uint8),
                #     )

                if world_rank == 0 and tb_every > 0 and step % tb_every == 0:
                    mem = torch.cuda.max_memory_allocated() / 1024**3
                    self.writer.add_scalar("train/loss", loss.item(), step)
                    self.writer.add_scalar("train/l1loss", l1loss.item(), step)
                    self.writer.add_scalar("train/ssimloss", ssimloss.item(), step)
                    self.writer.add_scalar("train/num_GS", len(self.splats["means"]), step)
                    self.writer.add_scalar("train/mem", mem, step)
                    self.writer.add_scalar("train/epoch", epoch, step)
                    # Running averages
                    self.writer.add_scalar("train/loss_avg", avg_loss, step)
                    self.writer.add_scalar("train/l1loss_avg", avg_l1, step)
                    self.writer.add_scalar("train/ssimloss_avg", avg_ssim, step)
                    if cfg.depth_loss:
                        self.writer.add_scalar("train/depthloss", depthloss.item(), step)
                        self.writer.add_scalar("train/depthloss_avg", avg_depth, step)
                    if cfg.ba_loss:
                        self.writer.add_scalar("train/baloss", baloss.item(), step)
                        self.writer.add_scalar("train/baloss_avg", avg_ba, step)
                        
                        # Gradient distribution analysis for BA observations
                        with torch.no_grad():
                            # Per-observation reprojection error statistics
                            reproj_err = self._ba_reproj_errors.flatten()  # [M]
                            per_obs_loss = self._ba_per_obs_loss.flatten()  # [M]
                            
                            # Reprojection error distribution
                            self.writer.add_scalar("ba/reproj_err_mean", reproj_err.mean().item(), step)
                            self.writer.add_scalar("ba/reproj_err_std", reproj_err.std().item(), step)
                            self.writer.add_scalar("ba/reproj_err_min", reproj_err.min().item(), step)
                            self.writer.add_scalar("ba/reproj_err_max", reproj_err.max().item(), step)
                            self.writer.add_scalar("ba/reproj_err_median", reproj_err.median().item(), step)
                            
                            # Percentiles (outlier detection)
                            self.writer.add_scalar("ba/reproj_err_p90", torch.quantile(reproj_err, 0.90).item(), step)
                            self.writer.add_scalar("ba/reproj_err_p95", torch.quantile(reproj_err, 0.95).item(), step)
                            self.writer.add_scalar("ba/reproj_err_p99", torch.quantile(reproj_err, 0.99).item(), step)
                            
                            # Per-observation loss distribution
                            self.writer.add_scalar("ba/per_obs_loss_mean", per_obs_loss.mean().item(), step)
                            self.writer.add_scalar("ba/per_obs_loss_std", per_obs_loss.std().item(), step)
                            self.writer.add_scalar("ba/per_obs_loss_max", per_obs_loss.max().item(), step)
                            
                            # Count of inliers/outliers (error > 2*threshold is likely outlier)
                            inlier_mask = reproj_err < cfg.ba_thres
                            outlier_mask = reproj_err > 2 * cfg.ba_thres
                            self.writer.add_scalar("ba/inlier_ratio", inlier_mask.float().mean().item(), step)
                            self.writer.add_scalar("ba/outlier_ratio", outlier_mask.float().mean().item(), step)
                            self.writer.add_scalar("ba/num_observations", reproj_err.shape[0], step)
                            
                            # Track point gradient analysis
                            if self.track_points_3d.grad is not None:
                                track_grad = self.track_points_3d.grad  # [N, 3]
                                grad_norms = torch.norm(track_grad, dim=-1)  # [N]
                                nonzero_mask = grad_norms > 0
                                if nonzero_mask.any():
                                    active_grads = grad_norms[nonzero_mask]
                                    self.writer.add_scalar("ba/grad_mean", active_grads.mean().item(), step)
                                    self.writer.add_scalar("ba/grad_std", active_grads.std().item(), step)
                                    self.writer.add_scalar("ba/grad_max", active_grads.max().item(), step)
                                    self.writer.add_scalar("ba/grad_min", active_grads.min().item(), step)
                                    self.writer.add_scalar("ba/num_active_tracks", nonzero_mask.sum().item(), step)
                            
                            # Histogram of reprojection errors (logged less frequently)
                            if step % (tb_every * 10) == 0:
                                self.writer.add_histogram("ba/reproj_err_hist", reproj_err.cpu(), step)
                                self.writer.add_histogram("ba/per_obs_loss_hist", per_obs_loss.cpu(), step)
                    if cfg.pose_opt:
                        with torch.no_grad():
                            if self.world_size > 1:
                                embeds = self.pose_adjust.module.embeds.weight
                            else:
                                embeds = self.pose_adjust.embeds.weight
                            dist = torch.norm(embeds[:, :3], dim=-1).mean()
                            self.writer.add_scalar("train/pose_dist", dist.item(), step)
                    if cfg.pose_noise:
                        self.writer.add_scalar("train/pose_noise_scale", noise_scale, step)
                    if cfg.ba_loss and self.ba_optimizers:
                        # Log current BA learning rate
                        ba_lr = self.ba_optimizers[0].param_groups[0]['lr']
                        self.writer.add_scalar("train/ba_lr", ba_lr, step)
                    if cfg.use_bilateral_grid:
                        self.writer.add_scalar("train/tvloss", tvloss.item(), step)
                    if cfg.tb_save_image:
                        canvas = torch.cat([pixels, colors], dim=2).detach().cpu().numpy()
                        canvas = canvas.reshape(-1, *canvas.shape[2:])
                        self.writer.add_image("train/render", canvas, step)
                    self.writer.flush()

                    if cfg.use_wandb:
                        log_dict = {
                            "train/loss": loss.item(),
                            "train/l1loss": l1loss.item(),
                            "train/ssimloss": ssimloss.item(),
                            "train/num_GS": len(self.splats["means"]),
                            "train/mem": mem,
                            "train/epoch": epoch,
                            "step": step,
                            # Running averages
                            "train/loss_avg": avg_loss,
                            "train/l1loss_avg": avg_l1,
                            "train/ssimloss_avg": avg_ssim,
                        }
                        if cfg.depth_loss:
                            log_dict["train/depthloss"] = depthloss.item()
                            log_dict["train/depthloss_avg"] = avg_depth
                        if cfg.ba_loss:
                            log_dict["train/baloss"] = baloss.item()
                            log_dict["train/baloss_avg"] = avg_ba
                            # BA gradient distribution analysis for wandb
                            with torch.no_grad():
                                reproj_err = self._ba_reproj_errors.flatten()
                                per_obs_loss = self._ba_per_obs_loss.flatten()
                                
                                log_dict.update({
                                    "ba/reproj_err_mean": reproj_err.mean().item(),
                                    "ba/reproj_err_std": reproj_err.std().item(),
                                    "ba/reproj_err_min": reproj_err.min().item(),
                                    "ba/reproj_err_max": reproj_err.max().item(),
                                    "ba/reproj_err_median": reproj_err.median().item(),
                                    "ba/reproj_err_p90": torch.quantile(reproj_err, 0.90).item(),
                                    "ba/reproj_err_p95": torch.quantile(reproj_err, 0.95).item(),
                                    "ba/reproj_err_p99": torch.quantile(reproj_err, 0.99).item(),
                                    "ba/per_obs_loss_mean": per_obs_loss.mean().item(),
                                    "ba/per_obs_loss_std": per_obs_loss.std().item(),
                                    "ba/per_obs_loss_max": per_obs_loss.max().item(),
                                    "ba/inlier_ratio": (reproj_err < cfg.ba_thres).float().mean().item(),
                                    "ba/outlier_ratio": (reproj_err > 2 * cfg.ba_thres).float().mean().item(),
                                    "ba/num_observations": reproj_err.shape[0],
                                })
                                
                                if self.track_points_3d.grad is not None:
                                    track_grad = self.track_points_3d.grad
                                    grad_norms = torch.norm(track_grad, dim=-1)
                                    nonzero_mask = grad_norms > 0
                                    if nonzero_mask.any():
                                        active_grads = grad_norms[nonzero_mask]
                                        log_dict.update({
                                            "ba/grad_mean": active_grads.mean().item(),
                                            "ba/grad_std": active_grads.std().item(),
                                            "ba/grad_max": active_grads.max().item(),
                                            "ba/grad_min": active_grads.min().item(),
                                            "ba/num_active_tracks": nonzero_mask.sum().item(),
                                        })
                                        
                                # Histograms for wandb (logged less frequently)
                                if step % (tb_every * 10) == 0:
                                    log_dict["ba/reproj_err_hist"] = wandb.Histogram(reproj_err.cpu().numpy())
                                    log_dict["ba/per_obs_loss_hist"] = wandb.Histogram(per_obs_loss.cpu().numpy())
                                    if self.track_points_3d.grad is not None and nonzero_mask.any():
                                        log_dict["ba/grad_hist"] = wandb.Histogram(active_grads.cpu().numpy())
                        if cfg.pose_opt:
                            with torch.no_grad():
                                if self.world_size > 1:
                                    embeds = self.pose_adjust.module.embeds.weight
                                else:
                                    embeds = self.pose_adjust.embeds.weight
                                dist = torch.norm(embeds[:, :3], dim=-1).mean()
                                log_dict["train/pose_dist"] = dist.item()
                        if cfg.pose_noise:
                            log_dict["train/pose_noise_scale"] = noise_scale
                        if cfg.ba_loss and self.ba_optimizers:
                            ba_lr = self.ba_optimizers[0].param_groups[0]['lr']
                            log_dict["train/ba_lr"] = ba_lr
                        if cfg.use_bilateral_grid:
                            log_dict["train/tvloss"] = tvloss.item()
                        if cfg.tb_save_image:
                            log_dict["train/render"] = wandb.Image(canvas)
                        wandb.log(log_dict, step=step)

                # save checkpoint before updating the model
                if step in save_steps or step == effective_max_steps - 1:
                    mem = torch.cuda.max_memory_allocated() / 1024**3
                    stats = {
                        "mem": mem,
                        "ellipse_time": time.time() - global_tic,
                        "num_GS": len(self.splats["means"]),
                    }
                    print("Step: ", step, stats)
                    with open(
                        f"{self.stats_dir}/train_step{step:04d}_rank{self.world_rank}.json",
                        "w",
                    ) as f:
                        json.dump(stats, f)
                    ckpt_data = {"step": step, "splats": self.splats.state_dict()}
                    if cfg.pose_opt:
                        if world_size > 1:
                            ckpt_data["pose_adjust"] = self.pose_adjust.module.state_dict()
                            ckpt_data["pose_adjust_test"] = self.pose_adjust_test.module.state_dict()
                        else:
                            ckpt_data["pose_adjust"] = self.pose_adjust.state_dict()
                            ckpt_data["pose_adjust_test"] = self.pose_adjust_test.state_dict()
                    if cfg.app_opt:
                        if world_size > 1:
                            ckpt_data["app_module"] = self.app_module.module.state_dict()
                        else:
                            ckpt_data["app_module"] = self.app_module.state_dict()
                    torch.save(
                        ckpt_data, f"{self.ckpt_dir}/ckpt_{step}_rank{self.world_rank}.pt"
                    )

                # Turn Gradients into Sparse Tensor before running optimizer
                if cfg.sparse_grad:
                    assert cfg.packed, "Sparse gradients only work with packed mode."
                    gaussian_ids = info["gaussian_ids"]
                    for k in self.splats.keys():
                        grad = self.splats[k].grad
                        if grad is None or grad.is_sparse:
                            continue
                        self.splats[k].grad = torch.sparse_coo_tensor(
                            indices=gaussian_ids[None],  # [1, nnz]
                            values=grad[gaussian_ids],  # [nnz, ...]
                            size=self.splats[k].size(),  # [N, ...]
                            is_coalesced=len(Ks) == 1,
                        )

                if cfg.visible_adam:
                    gaussian_cnt = self.splats.means.shape[0]
                    if cfg.packed:
                        visibility_mask = torch.zeros_like(
                            self.splats["opacities"], dtype=bool
                        )
                        visibility_mask.scatter_(0, info["gaussian_ids"], 1)
                    else:
                        visibility_mask = (info["radii"] > 0).any(0)

                # optimize
                for optimizer in self.optimizers.values():
                    if cfg.visible_adam:
                        optimizer.step(visibility_mask)
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                for optimizer in self.pose_optimizers:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                for optimizer in self.app_optimizers:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                for optimizer in self.bil_grid_optimizers:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                for optimizer in self.ba_optimizers:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                
                for scheduler in schedulers:
                    scheduler.step()

                if isinstance(self.cfg.strategy, ImportanceGuidedMCMCStrategy):
                    strategy = self.cfg.strategy
                    need_scores = (
                        step > strategy.refine_start_iter
                        and step < strategy.refine_stop_iter
                        and step % strategy.refine_every == 0
                    )
                    if need_scores:
                        importance_score, redundancy_score = self.compute_fastgs_scores(
                            sh_degree=sh_degree_to_use, num_views=cfg.fastgs_num_views
                        )
                        self.strategy_state["fastgs_importance"] = importance_score
                        self.strategy_state["fastgs_redundancy"] = redundancy_score
                        self.strategy_state["fastgs_pruning"] = redundancy_score
                    else:
                        self.strategy_state["fastgs_importance"] = None
                        self.strategy_state["fastgs_redundancy"] = None
                        self.strategy_state["fastgs_pruning"] = None

                # Run post-backward steps after backward and optimizer
                if isinstance(self.cfg.strategy, DefaultStrategy):
                    self.cfg.strategy.step_post_backward(
                        params=self.splats,
                        optimizers=self.optimizers,
                        state=self.strategy_state,
                        step=step,
                        info=info,
                        packed=cfg.packed,
                    )
                elif isinstance(self.cfg.strategy, (MCMCStrategy, ImportanceGuidedMCMCStrategy)):
                    self.cfg.strategy.step_post_backward(
                        params=self.splats,
                        optimizers=self.optimizers,
                        state=self.strategy_state,
                        step=step,
                        info=info,
                        lr=schedulers[0].get_last_lr()[0],
                    )
                else:
                    raise ValueError(f"Unknown strategy type: {type(self.cfg.strategy)}")

                # eval the full set (both train and val)
                if step in eval_steps:
                    self.eval(step, stage="val")
                    # self.eval(step, stage="train")
                    self.render_traj(step)

                # run compression
                if cfg.compression is not None and step in eval_steps:
                    self.run_compression(step=step)

                if not cfg.disable_viewer:
                    self.viewer.lock.release()
                    num_train_steps_per_sec = 1.0 / (time.time() - tic)
                    num_train_rays_per_sec = (
                        num_train_rays_per_step * num_train_steps_per_sec
                    )
                    # Update the viewer state.
                    self.viewer.state.num_train_rays_per_sec = num_train_rays_per_sec
                    # Update the scene.
                    self.viewer.update(step, num_train_rays_per_step)

                # Update step counter and progress bar
                step += 1
                pbar.update(1)

            # Test pose optimization: iterate through all test views after training epoch
            # Freeze GS, only optimize test camera poses
            if cfg.pose_opt:
                sh_degree_to_use = min(step // sh_degree_interval, cfg.sh_degree)
                for test_data in testloader:
                    test_camtoworlds = test_camtoworlds_gt = test_data["camtoworld"].to(device)
                    test_Ks = test_data["K"].to(device)
                    test_pixels = test_data["image"].to(device) / 255.0
                    test_image_ids = test_data["image_id"].to(device)
                    test_abs_indices = test_data["index"].to(device) if "index" in test_data else None
                    test_masks = test_data["mask"].to(device) if "mask" in test_data else None
                    test_height, test_width = test_pixels.shape[1:3]

                    # Apply pose adjustment for test images
                    test_camtoworlds = self.pose_adjust_test(test_camtoworlds, test_image_ids)

                    # Fix the pose of camera 0 (same as training)
                    if test_abs_indices is not None:
                        mask = (test_abs_indices == 0)
                        if mask.any():
                            test_camtoworlds = torch.where(mask[..., None, None], test_camtoworlds_gt, test_camtoworlds)

                    # Forward pass with frozen GS parameters
                    test_renders, test_alphas, _ = self.rasterize_splats(
                        camtoworlds=test_camtoworlds,
                        Ks=test_Ks,
                        width=test_width,
                        height=test_height,
                        sh_degree=sh_degree_to_use,
                        near_plane=cfg.near_plane,
                        far_plane=cfg.far_plane,
                        image_ids=test_image_ids,
                        render_mode="RGB",
                        masks=test_masks,
                        freeze_gs=True,  # Freeze GS parameters
                    )
                    test_colors = test_renders[..., 0:3] if test_renders.shape[-1] == 4 else test_renders

                    # Compute loss for test pose optimization
                    test_l1loss = F.l1_loss(test_colors, test_pixels)
                    test_ssimloss = 1.0 - fused_ssim(
                        test_colors.permute(0, 3, 1, 2), test_pixels.permute(0, 3, 1, 2), padding="valid"
                    )
                    test_loss = test_l1loss * (1.0 - cfg.ssim_lambda) + test_ssimloss * cfg.ssim_lambda

                    # BA loss for test pose optimization (keep track points optimizable, freeze GS only)
                    if cfg.ba_loss and "point_indices" in test_data:
                        test_baloss, _, _, _ = self.compute_ba_loss(
                            test_data, test_camtoworlds, test_Ks, freeze_points=False
                        )
                        test_loss = test_loss + test_baloss * cfg.ba_lambda

                    # Backward and optimize test pose + BA keypoints (GS remains frozen)
                    test_loss.backward()
                    for optimizer in self.pose_optimizers_test:
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                    for optimizer in self.ba_optimizers:
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)

                # Log test pose optimization at end of epoch
                if world_rank == 0 and tb_every > 0:
                    self.writer.add_scalar("test_pose/loss", test_loss.item(), step)
                    self.writer.add_scalar("test_pose/l1loss", test_l1loss.item(), step)
                    self.writer.add_scalar("test_pose/ssimloss", test_ssimloss.item(), step)
                    if cfg.ba_loss:
                        self.writer.add_scalar("test_pose/baloss", test_baloss.item(), step)
                    with torch.no_grad():
                        if self.world_size > 1:
                            test_embeds = self.pose_adjust_test.module.embeds.weight
                        else:
                            test_embeds = self.pose_adjust_test.embeds.weight
                        test_dist = torch.norm(test_embeds[:, :3], dim=-1).mean()
                        self.writer.add_scalar("test_pose/pose_dist", test_dist.item(), step)
                    
                    if cfg.use_wandb:
                        log_dict = {
                            "test_pose/loss": test_loss.item(),
                            "test_pose/l1loss": test_l1loss.item(),
                            "test_pose/ssimloss": test_ssimloss.item(),
                            "test_pose/pose_dist": test_dist.item(),
                            "test_pose/step": step,
                            "test_pose/epoch": epoch,
                        }
                        if cfg.ba_loss:
                            log_dict["test_pose/baloss"] = test_baloss.item()
                        wandb.log(log_dict, commit=False)

        pbar.close()

    @torch.no_grad()
    def eval(self, step: int, stage: str = "val", dataset=None):
        """Entry for evaluation."""
        print(f"Running evaluation on {stage} set...")
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank
        world_size = self.world_size

        # Use provided dataset or default to valset
        if dataset is None:
            dataset = self.valset if stage == "val" else self.trainset
        
        evalloader = torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=1
        )
        ellipse_time = 0
        metrics = defaultdict(list)
        for i, data in enumerate(evalloader):
            camtoworlds = camtoworlds_gt = data["camtoworld"].to(device)
            Ks = data["K"].to(device)
            pixels = data["image"].to(device) / 255.0
            masks = data["mask"].to(device) if "mask" in data else None
            image_ids = data["image_id"].to(device) if "image_id" in data else None
            abs_indices = data["index"].to(device) if "index" in data else None
            height, width = pixels.shape[1:3]

            # Apply pose adjustment if enabled (must match training)
            # Use pose_adjust for train images, pose_adjust_test for val images
            if cfg.pose_opt and hasattr(self, 'pose_adjust'):
                if stage == "val" and hasattr(self, 'pose_adjust_test'):
                    camtoworlds = self.pose_adjust_test(camtoworlds, image_ids)
                else:
                    camtoworlds = self.pose_adjust(camtoworlds, image_ids)
                
                # Fix the pose of camera 0 (same as training)
                if abs_indices is not None:
                    mask = (abs_indices == 0)
                    if mask.any():
                        camtoworlds = torch.where(mask[..., None, None], camtoworlds_gt, camtoworlds)

            torch.cuda.synchronize()
            tic = time.time()
            colors, _, _ = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                masks=masks,
                image_ids=image_ids,
            )  # [1, H, W, 3]
            torch.cuda.synchronize()
            ellipse_time += time.time() - tic

            # Apply bilateral grid if enabled (must match training)
            if cfg.use_bilateral_grid and hasattr(self, 'bil_grids'):
                grid_y, grid_x = torch.meshgrid(
                    (torch.arange(height, device=self.device) + 0.5) / height,
                    (torch.arange(width, device=self.device) + 0.5) / width,
                    indexing="ij",
                )
                grid_xy = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
                colors = slice(self.bil_grids, grid_xy, colors, image_ids)["rgb"]

            colors = torch.clamp(colors, 0.0, 1.0)
            canvas_list = [pixels, colors]

            if world_rank == 0:
                # write images
                canvas = torch.cat(canvas_list, dim=2).squeeze(0).cpu().numpy()
                canvas = (canvas * 255).astype(np.uint8)
                imageio.imwrite(
                    f"{self.render_dir}/{stage}_step{step}_{i:04d}.png",
                    canvas,
                )

                pixels_p = pixels.permute(0, 3, 1, 2)  # [1, 3, H, W]
                colors_p = colors.permute(0, 3, 1, 2)  # [1, 3, H, W]
                metrics["psnr"].append(self.psnr(colors_p, pixels_p))
                metrics["ssim"].append(self.ssim(colors_p, pixels_p))
                metrics["lpips"].append(self.lpips(colors_p, pixels_p))
                if cfg.use_bilateral_grid:
                    cc_colors = color_correct(colors, pixels)
                    cc_colors_p = cc_colors.permute(0, 3, 1, 2)  # [1, 3, H, W]
                    metrics["cc_psnr"].append(self.psnr(cc_colors_p, pixels_p))

        if world_rank == 0:
            ellipse_time /= len(evalloader)

            stats = {k: torch.stack(v).mean().item() for k, v in metrics.items()}
            stats.update(
                {
                    "ellipse_time": ellipse_time,
                    "num_GS": len(self.splats["means"]),
                }
            )
            print(
                f"PSNR: {stats['psnr']:.3f}, SSIM: {stats['ssim']:.4f}, LPIPS: {stats['lpips']:.3f} "
                f"Time: {stats['ellipse_time']:.3f}s/image "
                f"Number of GS: {stats['num_GS']}"
            )
            # save stats as json
            with open(f"{self.stats_dir}/{stage}_step{step:04d}.json", "w") as f:
                json.dump(stats, f)
            # save stats to tensorboard
            for k, v in stats.items():
                self.writer.add_scalar(f"{stage}/{k}", v, step)
            self.writer.flush()

            if cfg.use_wandb:
                wandb.log({f"{stage}/{k}": v for k, v in stats.items()}, step=step)

    @torch.no_grad()
    def render_traj(self, step: int):
        """Entry for trajectory rendering."""
        print("Running trajectory rendering...")
        cfg = self.cfg
        device = self.device

        camtoworlds_all = self.parser.camtoworlds[5:-5]
        if cfg.render_traj_path == "interp":
            camtoworlds_all = generate_interpolated_path(
                camtoworlds_all, 1
            )  # [N, 3, 4]
        elif cfg.render_traj_path == "ellipse":
            height = camtoworlds_all[:, 2, 3].mean()
            camtoworlds_all = generate_ellipse_path_z(
                camtoworlds_all, height=height
            )  # [N, 3, 4]
        elif cfg.render_traj_path == "spiral":
            camtoworlds_all = generate_spiral_path(
                camtoworlds_all,
                bounds=self.parser.bounds * self.scene_scale,
                spiral_scale_r=self.parser.extconf["spiral_radius_scale"],
            )
        else:
            raise ValueError(
                f"Render trajectory type not supported: {cfg.render_traj_path}"
            )

        camtoworlds_all = np.concatenate(
            [
                camtoworlds_all,
                np.repeat(
                    np.array([[[0.0, 0.0, 0.0, 1.0]]]), len(camtoworlds_all), axis=0
                ),
            ],
            axis=1,
        )  # [N, 4, 4]

        camtoworlds_all = torch.from_numpy(camtoworlds_all).float().to(device)
        K = torch.from_numpy(list(self.parser.Ks_dict.values())[0]).float().to(device)
        width, height = list(self.parser.imsize_dict.values())[0]

        # save to video
        video_dir = f"{cfg.result_dir}/videos"
        os.makedirs(video_dir, exist_ok=True)
        writer = imageio.get_writer(f"{video_dir}/traj_{step}.mp4", fps=30)
        for i in tqdm.trange(len(camtoworlds_all), desc="Rendering trajectory"):
            camtoworlds = camtoworlds_all[i : i + 1]
            Ks = K[None]

            renders, _, _ = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                render_mode="RGB+ED",
            )  # [1, H, W, 4]
            colors = torch.clamp(renders[..., 0:3], 0.0, 1.0)  # [1, H, W, 3]
            depths = renders[..., 3:4]  # [1, H, W, 1]
            depths = (depths - depths.min()) / (depths.max() - depths.min())
            canvas_list = [colors, depths.repeat(1, 1, 1, 3)]

            # write images
            canvas = torch.cat(canvas_list, dim=2).squeeze(0).cpu().numpy()
            canvas = (canvas * 255).astype(np.uint8)
            writer.append_data(canvas)
        writer.close()
        print(f"Video saved to {video_dir}/traj_{step}.mp4")

    @torch.no_grad()
    def run_compression(self, step: int):
        """Entry for running compression."""
        print("Running compression...")
        world_rank = self.world_rank

        compress_dir = f"{cfg.result_dir}/compression/rank{world_rank}"
        os.makedirs(compress_dir, exist_ok=True)

        self.compression_method.compress(compress_dir, self.splats)

        # evaluate compression
        splats_c = self.compression_method.decompress(compress_dir)
        for k in splats_c.keys():
            self.splats[k].data = splats_c[k].to(self.device)
        self.eval(step=step, stage="compress")

    @torch.no_grad()
    def _viewer_render_fn(
        self, camera_state: nerfview.CameraState, img_wh: Tuple[int, int]
    ):
        """Callable function for the viewer."""
        W, H = img_wh
        c2w = camera_state.c2w
        K = camera_state.get_K(img_wh)
        c2w = torch.from_numpy(c2w).float().to(self.device)
        K = torch.from_numpy(K).float().to(self.device)

        render_colors, _, _ = self.rasterize_splats(
            camtoworlds=c2w[None],
            Ks=K[None],
            width=W,
            height=H,
            sh_degree=self.cfg.sh_degree,  # active all SH degrees
            radius_clip=3.0,  # skip GSs that have small image radius (in pixels)
        )  # [1, H, W, 3]
        return render_colors[0].cpu().numpy()


def main(local_rank: int, world_rank, world_size: int, cfg: Config):
    if world_size > 1 and not cfg.disable_viewer:
        cfg.disable_viewer = True
        if world_rank == 0:
            print("Viewer is disabled in distributed training.")

    runner = Runner(local_rank, world_rank, world_size, cfg)

    if cfg.ckpt is not None:
        # run eval only
        ckpts = [
            torch.load(file, map_location=runner.device, weights_only=True)
            for file in cfg.ckpt
        ]
        for k in runner.splats.keys():
            runner.splats[k].data = torch.cat([ckpt["splats"][k] for ckpt in ckpts])
        step = ckpts[0]["step"]
        runner.eval(step=step, stage="val")
        runner.eval(step=step, stage="train")
        runner.render_traj(step=step)
        if cfg.compression is not None:
            runner.run_compression(step=step)
    else:
        runner.train()

    if not cfg.disable_viewer and not cfg.close_viewer_after_training:
        print("Viewer running... Ctrl+C to exit.")
        time.sleep(1000000)


if __name__ == "__main__":
    """
    Usage:

    ```bash
    # Single GPU training
    CUDA_VISIBLE_DEVICES=0 python simple_trainer.py default

    # Distributed training on 4 GPUs: Effectively 4x batch size so run 4x less epochs.
    CUDA_VISIBLE_DEVICES=0,1,2,3 python simple_trainer.py default --epochs_scaler 0.25

    """

    # Config objects we can choose between.
    # Each is a tuple of (CLI description, config object).
    configs = {
        "default": (
            "Gaussian splatting training using densification heuristics from the original paper.",
            Config(
                strategy=DefaultStrategy(verbose=True),
                pose_opt=True,
                ba_loss=True,
            ),
        ),
        "mcmc": (
            "Gaussian splatting training using densification from the paper '3D Gaussian Splatting as Markov Chain Monte Carlo'.",
            Config(
                init_opa=0.5,
                init_scale=0.1,
                opacity_reg=0.01,
                scale_reg=0.01,
                strategy=MCMCStrategy(verbose=True),
                pose_opt=True,
                ba_loss=True,
            ),
        ),
        "mcmc_importance": (
            "Gaussian splatting training using importance-guided MCMC proposals.",
            Config(
                init_opa=0.5,
                init_scale=0.1,
                opacity_reg=0.01,
                scale_reg=0.01,
                strategy=ImportanceGuidedMCMCStrategy(verbose=True),
                pose_opt=True,
                ba_loss=True,
            ),
        ),
    }
    cfg = tyro.extras.overridable_config_cli(configs)
    cfg.adjust_epochs(cfg.epochs_scaler)
    cli(main, cfg, verbose=True)