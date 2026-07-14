from loguru import logger
from typing import Tuple, Union
import typing
import torch
import torch.nn as nn
import torch.nn.functional as F

from fastmap.container import Matches, Cameras, MatchesData
from fastmap.fundamental import estimate_fundamental
from fastmap.utils import to_homogeneous, ConvergenceManager


def undistort(xy: torch.Tensor, alpha: Union[float, torch.Tensor]):
    """Assume the distortion model is the one-parameter division model. See <Radial Distortion Self-Calibration>
    Args:
        xy: torch.Tensor float (num_point_pairs, 2), the xy coordinates
        alpha: Union[float, torch.Tensor], the distortion parameter
    Returns:
        torch.Tensor float (num_point_pairs, 2), the undistorted xy coordinates
    """
    assert xy.shape[-1] == 2
    if not isinstance(alpha, torch.Tensor):
        alpha = torch.tensor(alpha).to(xy)
    alpha = alpha.expand(xy.shape[:-1])  # (...)
    r2 = torch.sum(xy**2, dim=-1)  # (...,)
    return xy / (1.0 + alpha * r2).unsqueeze(-1)


@torch.no_grad()
def alpha_to_k1(
    alpha: torch.Tensor,
    T: torch.Tensor,
    cameras: Cameras,
    num_samples: int = 10000,
    lr: float = 0.0001,
    max_iters: int = 50000,
    log_interval: int = 100,
):
    """Convert the division undistortion parameter alpha to k1 parameter of the Brown–Conrady variant used in OpenCV and COLMAP.
    Args:
        alpha: torch.Tensor float (num_cameras,), the division model parameter alpha
        T: torch.Tensor float (num_cameras, 3, 3), the transformation matrix after which the distortion is estimated
        camera: Cameras container
        num_samples: int, the number of samples for optimization
        lr: float, the learning rate for optimization
        max_iters: int, the maximum number of iterations for optimization
        log_interval: int, the interval for logging in number of iterations
    Returns:
        k1: torch.Tensor float (num_cameras,), the k1 parameter in pixel space
    """
    # get information
    device, dtype = alpha.device, alpha.dtype
    num_cameras = alpha.shape[0]
    assert T.shape == (num_cameras, 3, 3)
    assert cameras.calibrated  # make sure the focal lengths are known

    # define the distortion function for one-parameter Brown–Conrady model
    def distort_brown_conrady(_xy, _k1):
        _r2 = torch.sum(_xy**2, dim=-1)  # (...,)
        return _xy * (1.0 + _k1 * _r2).unsqueeze(-1)

    # sample data for optimization
    xy_distorted = (
        torch.rand(num_cameras, num_samples, 2, device=device, dtype=dtype) * 2.0 - 1.0
    )  # (num_cameras, num_samples, 2)
    xy_undistorted = undistort(
        xy=xy_distorted, alpha=alpha[:, None].expand(xy_distorted.shape[:2])
    )  # (num_cameras, num_samples, 2)
    xy_target = xy_distorted  # (num_cameras, num_samples, 2)
    # del xy_pixel_distorted

    # initialize parameters
    k1 = torch.zeros(num_cameras, device=device, dtype=dtype)  # (num_cameras,)
    k1 = nn.Parameter(k1, requires_grad=True)  # (num_cameras,)

    # build optimizer
    optimizer = torch.optim.Adam([k1], lr=lr)  # (num_cameras,)

    # build convergence manager
    convergence_manager = ConvergenceManager(
        warmup_steps=10, decay=0.9, convergence_window=100
    )
    convergence_manager.start()

    # loop
    with torch.enable_grad():
        for iter_idx in range(max_iters):
            xy_pred = distort_brown_conrady(
                _xy=xy_undistorted, _k1=k1[:, None].expand(xy_undistorted.shape[:2])
            )  # (num_cameras, num_samples, 2)
            loss = F.l1_loss(xy_pred, xy_target)  # ()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # check convergence
            moving_loss, if_converged = convergence_manager.step(
                step=iter_idx, loss=loss
            )  # float, bool
            if if_converged:
                logger.info(
                    f"Converged at iteration {iter_idx+1} with loss {moving_loss} and mean k1 {k1.mean().item()}"
                )
                break

            if iter_idx % log_interval == 0:
                logger.info(
                    f"[Iter {iter_idx+1}] loss: {loss.item()}, mean k1: {k1.mean().item()}"
                )

    # rescale k1 with T and focal
    k1 = k1.data  # (num_cameras,)
    assert torch.allclose(T[:, 0, 0], T[:, 1, 1])  # make sure it is isotropic
    k1 *= (T[:, 0, 0] * cameras.focal) ** 2.0  # (num_cameras,)
    logger.info(f"Final k1 in pixel space: {k1}")

    # return
    return k1


@torch.no_grad()
def compute_epipolar_error(
    matches_data: MatchesData,
    fundamental: torch.Tensor,
) -> torch.Tensor:
    """Compute the epipolar error for point pairs in the MatchesData container.
    Args:
        matches_data: MatchesData
        fundamental: torch.Tensor float (num_image_pairs, 3, 3), the fundamental matrix for all image pair (might contain invalid values in slots not related to the image pairs in matches_data)
    Returns:
        error: torch.Tensor float (num_point_pairs,), the epipolar error for each point pair in matches_data
    """
    # to homogeneous
    xy_homo1 = to_homogeneous(matches_data.xy1)  # (num_point_pairs, 3)
    xy_homo2 = to_homogeneous(matches_data.xy2)  # (num_point_pairs, 3)
    xy_homo1 = F.normalize(xy_homo1, p=2, dim=-1)  # (num_point_pairs, 3)
    xy_homo2 = F.normalize(xy_homo2, p=2, dim=-1)  # (num_point_pairs, 3)

    # compute error
    error = torch.einsum(
        "bi,bij,bj->b",
        xy_homo2,
        fundamental[matches_data.image_pair_idx],
        xy_homo1,
    ).abs()  # (num_point_pairs,)

    return error


@torch.no_grad()
def estimate_distortion(
    matches: Matches,
    cameras: Cameras,
    num_levels: int = 3,
    num_samples: int = 10,
    alpha_range: Union[Tuple[float, float], torch.Tensor] = (-0.5, 0.2),
):
    """Estimate the common distortion parameter alpha from the matches.

    Args:
        matches: Matches
        cameras: Cameras
        num_levels: int, the number of levels for the grid search
        num_samples: int, the number of samples per level
        alpha_range: Tuple[float, float], the range of the distortion parameter
    Returns:
        alpha: torch.Tensor float (num_cameras,), the distortion parameter for the common division distortion model.
        T: torch.Tensor float (num_cameras, 3, 3), the transformation matrix after which the distortion is estimated
    """
    # get some useful information
    device = matches.xy.device
    dtype = matches.xy.dtype
    num_images = (
        max(matches.image_idx1.max().item(), matches.image_idx2.max().item()) + 1
    )
    num_images = typing.cast(int, num_images)
    assert torch.all(matches.image_idx1 < matches.image_idx2)
    num_cameras = cameras.num_cameras

    # get the matrix for transforming xy so that the principal point is at (0, 0) and the norm is in a reasonable range
    xy_center = torch.stack([cameras.cx, cameras.cy], dim=-1)  # (num_cameras, 2)
    xy_scale = 1.0 / xy_center.norm(
        dim=-1
    )  # (num_cameras,) assuming the principal point is at the center
    T = (
        torch.eye(3, device=device, dtype=dtype).expand(num_cameras, -1, -1).clone()
    )  # (num_cameras, 3, 3)
    T[:, 0, 0] = xy_scale
    T[:, 1, 1] = xy_scale
    T[:, 0, 2] = -xy_center[:, 0] * xy_scale
    T[:, 1, 2] = -xy_center[:, 1] * xy_scale

    # initialize the mask for cameras. True if the camera is already estimated.
    camera_mask = torch.zeros(
        num_cameras, device=device, dtype=torch.bool
    )  # (num_cameras,)

    # initialize the mask for camera pairs. True if the pair is ready to be used for focal estimation.
    camera_pair_mask = torch.eye(
        num_cameras, device=device, dtype=torch.bool
    )  # (num_cameras, num_cameras)

    # get the count of camera pairs.
    camera_pair_count = torch.zeros(
        num_cameras * num_cameras, device=device, dtype=torch.long
    )  # (num_cameras * num_cameras)
    camera_pair_count.scatter_reduce_(
        dim=0,
        index=cameras.camera_idx[matches.image_idx1] * num_cameras
        + cameras.camera_idx[matches.image_idx2],
        src=torch.ones_like(matches.image_idx1),
        reduce="sum",
    )  # (num_cameras * num_cameras)
    camera_pair_count = camera_pair_count.view(
        num_cameras, num_cameras
    )  # (num_cameras, num_cameras)

    # define a function to get the next camera to estimate focal
    def _next_camera_idx():
        num_pairs_available = (camera_pair_count * camera_pair_mask.long()).sum(
            dim=0
        ) + (camera_pair_count * camera_pair_mask.long()).sum(
            dim=1
        )  # (num_cameras,)
        num_pairs_available *= 1 - camera_mask.long()  # (num_cameras,)
        return typing.cast(int, num_pairs_available.argmax().item())

    # initialize the distortion parameter (the value is only valid when the camera is estimated)
    alpha = torch.nan + torch.zeros(
        num_cameras, device=device, dtype=dtype
    )  # (num_cameras,)

    # estimate the distortion for different cameras sequentially
    for _ in range(num_cameras):
        # get the camera idx
        current_camera_idx = _next_camera_idx()

        # get mask of valid image pairs and point pairs
        image_pair_mask = (
            camera_pair_mask[
                cameras.camera_idx[matches.image_idx1],
                cameras.camera_idx[matches.image_idx2],
            ]
            & (
                (cameras.camera_idx[matches.image_idx1] == current_camera_idx)
                | (cameras.camera_idx[matches.image_idx2] == current_camera_idx)
            )
            & matches.is_epipolar
        )  # (num_image_pairs,)
        if not torch.any(image_pair_mask):
            raise Exception(
                f"There is no valid image pair for estimating distortion of camera {current_camera_idx}."
            )
        logger.info(
            f"[Camera {current_camera_idx}] Using {image_pair_mask.long().sum().item()} image pairs for estimating distortion."
        )

        # intialize the alpha range for grid search
        current_alpha_range = alpha_range

        # initialize the best alpha to be None
        best_alpha = None

        # loop over the levels
        for level_idx in range(num_levels):
            logger.info(
                f"[Camera {current_camera_idx}] Level {level_idx+1} / {num_levels}"
            )
            # get all the alpha candidates
            alpha_candidates = torch.linspace(
                current_alpha_range[0],
                current_alpha_range[1],
                num_samples,
                device=device,
                dtype=dtype,
            )  # (num_samples,)

            # initialize error accumulation
            error_sum = torch.zeros(
                num_samples, device=device, dtype=dtype
            )  # (num_samples,)

            # loop over batches
            for matches_data in matches.query(image_pair_mask):

                # get camera idx
                batch_camera_idx1 = cameras.camera_idx[
                    matches_data.image_idx1
                ]  # (num_batch_point_pairs,)
                batch_camera_idx2 = cameras.camera_idx[
                    matches_data.image_idx2
                ]  # (num_batch_point_pairs,)

                # get transformed xy
                batch_xy_homo1 = to_homogeneous(
                    matches_data.xy1
                )  # (num_batch_point_pairs, 3)
                batch_xy_homo2 = to_homogeneous(
                    matches_data.xy2
                )  # (num_batch_point_pairs, 3)
                batch_xy_homo1 = torch.einsum(
                    "bij,bj->bi", T[batch_camera_idx1], batch_xy_homo1
                )  # (num_batch_point_pairs, 3)
                batch_xy_homo2 = torch.einsum(
                    "bij,bj->bi", T[batch_camera_idx2], batch_xy_homo2
                )  # (num_batch_point_pairs, 3)
                assert torch.all((batch_xy_homo1[..., -1] - 1.0).abs() < 1e-6)
                assert torch.all((batch_xy_homo2[..., -1] - 1.0).abs() < 1e-6)
                batch_transformed_xy1 = (
                    batch_xy_homo1[:, :2] / batch_xy_homo1[:, 2:3]
                )  # (B, 2)
                batch_transformed_xy2 = (
                    batch_xy_homo2[:, :2] / batch_xy_homo2[:, 2:3]
                )  # (B, 2)

                # get the current alphas for the point pairs (the values of those of current camera will be updated with the samples)
                alpha1 = alpha[batch_camera_idx1].clone()  # (num_batch_point_pairs,)
                alpha2 = alpha[batch_camera_idx2].clone()  # (num_batch_point_pairs,)

                # loop over alpha samples
                for sample_idx in range(num_samples):
                    # update the alpha with the new candidate
                    alpha1[batch_camera_idx1 == current_camera_idx] = alpha_candidates[
                        sample_idx
                    ]
                    alpha2[batch_camera_idx2 == current_camera_idx] = alpha_candidates[
                        sample_idx
                    ]

                    # undistort and replace the xy
                    batch_undistorted_xy1 = undistort(
                        xy=batch_transformed_xy1, alpha=alpha1
                    )  # (num_batch_point_pairs, 2)
                    batch_undistorted_xy2 = undistort(
                        xy=batch_transformed_xy2, alpha=alpha2
                    )  # (num_batch_point_pairs, 2)
                    matches_data.xy1 = batch_undistorted_xy1
                    matches_data.xy2 = batch_undistorted_xy2

                    # estimate the fundamental matrix (slots for image pairs not in this batch of data will be set to nan)
                    batch_fundamental = estimate_fundamental(
                        matches_data=matches_data,
                    )  # (num_image_pairs, 4, 4)

                    # compute per image pair error
                    batch_point_pair_error = compute_epipolar_error(
                        matches_data=matches_data,
                        fundamental=batch_fundamental,
                    )  # (num_batch_point_pairs,)

                    # accumulate error
                    error_sum[sample_idx] += batch_point_pair_error.sum().item()

            # find the best idx
            assert not torch.any(torch.isnan(error_sum))
            best_idx = error_sum.argmin().item()
            best_idx = typing.cast(int, best_idx)

            # find the current best alpha
            best_alpha = alpha_candidates[best_idx].item()
            best_alpha = typing.cast(float, best_alpha)

            # find the new alpha range
            min_idx = max(best_idx - 1, 0)
            max_idx = min(best_idx + 1, num_samples - 1)
            current_alpha_range = (
                alpha_candidates[min_idx].item(),
                alpha_candidates[max_idx].item(),
            )

        # update the alpha
        assert best_alpha is not None
        logger.info(f"[Camera {current_camera_idx}] Found alpha: {best_alpha}.")
        alpha[current_camera_idx] = best_alpha

        # indicate the camera is estimated
        camera_mask[current_camera_idx] = True
        camera_pair_mask[current_camera_idx] = True
        camera_pair_mask[:, current_camera_idx] = True

    # make sure all the alphas are estimated
    assert torch.all(~torch.isnan(alpha)), "Not all the alphas are estimated."

    # return
    return alpha, T


@torch.no_grad()
def undistort_matches(
    matches: Matches, cameras: Cameras, alpha: torch.Tensor, T: torch.Tensor
):
    """Undistort the xy coordinates. After this we need to re-estimate the fundamental and homography matrices, but this is not done in this function.

    Args:
        matches: Matches
        alpha: torch.Tensor float (num_cameras,), the distortion parameter for the common division distortion model.
        T: torch.Tensor float (num_cameras, 3, 3), the transformation matrix after which the undistortion is applied
    """
    ##### Transform xy and undistort #####
    # transform
    xy_homo = to_homogeneous(matches.xy).clone()  # (num_images, num_keypoints, 3)
    for camera_idx in range(cameras.num_cameras):
        xy_homo[cameras.camera_idx == camera_idx] = torch.einsum(
            "ij,...j->...i", T[camera_idx], xy_homo[cameras.camera_idx == camera_idx]
        )  # (num_images, num_keypoints, 3)
    xy = xy_homo[..., :-1] / xy_homo[..., -1:]  # (num_images, num_keypoints, 2)
    del xy_homo

    # undistort
    for camera_idx in range(cameras.num_cameras):
        xy[cameras.camera_idx == camera_idx] = undistort(
            xy=xy[cameras.camera_idx == camera_idx], alpha=alpha[camera_idx].item()
        )  # (num_images, num_keypoints, 2)

    # transform back
    xy_homo = to_homogeneous(xy).clone()  # (num_images, num_keypoints, 3)
    for camera_idx in range(cameras.num_cameras):
        xy_homo[cameras.camera_idx == camera_idx] = torch.einsum(
            "ij,...j->...i",
            T[camera_idx].inverse(),
            xy_homo[cameras.camera_idx == camera_idx],
        )  # (num_images, num_keypoints, 3)
    xy = xy_homo[..., :-1] / xy_homo[..., -1:]  # (num_images, num_keypoints, 2)
    del xy_homo

    # write to matches
    matches.xy = xy
    del xy, T
