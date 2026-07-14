from loguru import logger
import typing
import torch

from fastmap.container import Matches, Cameras
from fastmap.utils import to_homogeneous


@torch.no_grad()
def estimate_focals(
    cameras: Cameras,
    matches: Matches,
    min_fov: float = 20.0,
    max_fov: float = 160.0,
    num_samples: int = 100,
    std: float = 0.01,
):
    """Estimate the focal lengths from matches and modify the cameras container in place.
    Args:
        cameras: Cameras, cameras container
        matches: Matches, matches container
        min_fov: float, minimum horizontal fov
        max_fov: float, maximum horizontal fov
        num_samples: int, number of samples for fov estimation
        std: float, standard deviation for the votes
    Returns:
        In place update of the cameras container.
    """
    # get some info
    device, dtype = matches.xy.device, matches.xy.dtype
    num_cameras = cameras.num_cameras

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

    # estimate focal for each camera sequentially
    for _ in range(cameras.num_cameras):
        # get next camera idx
        current_camera_idx = _next_camera_idx()

        # get the image pair mask for focal estimation
        fundamental_mask = (
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

        # get the fundamental matrix
        fundamental = matches.matrix[fundamental_mask]  # (num_fundamental, 3, 3)
        num_fundamental = fundamental.shape[0]
        logger.info(
            f"[Camera {current_camera_idx}] Using {len(fundamental)} fundamental matrices for estimating focal..."
        )

        # get the image idx
        image_idx1 = matches.image_idx1[fundamental_mask]  # (num_fundamental,)
        image_idx2 = matches.image_idx2[fundamental_mask]  # (num_fundamental,)

        # get the camera idx for the pairs
        camera_idx1 = cameras.camera_idx[image_idx1]  # (num_fundamental,)
        camera_idx2 = cameras.camera_idx[image_idx2]  # (num_fundamental,)
        assert torch.all(
            (camera_idx1 == current_camera_idx) | (camera_idx2 == current_camera_idx)
        )

        # get the current focals and principal points for the image pairs
        focal1 = cameras.focal[camera_idx1]  # (num_fundamental,)
        focal2 = cameras.focal[camera_idx2]  # (num_fundamental,)
        cx1 = cameras.cx[camera_idx1]  # (num_fundamental,)
        cx2 = cameras.cx[camera_idx2]  # (num_fundamental,)
        cy1 = cameras.cy[camera_idx1]  # (num_fundamental,)
        cy2 = cameras.cy[camera_idx2]  # (num_fundamental,)

        # convert to intrinsics matrix (focals of the current camera will be replaced by samples later)
        K1 = (
            torch.eye(3, device=device, dtype=dtype)
            .expand(num_fundamental, 3, 3)
            .clone()
        )  # (num_fundamental, 3, 3)
        K1[:, 0, 0] = focal1
        K1[:, 1, 1] = focal1
        K1[:, 0, 2] = cx1
        K1[:, 1, 2] = cy1
        K2 = (
            torch.eye(3, device=device, dtype=dtype)
            .expand(num_fundamental, 3, 3)
            .clone()
        )  # (num_fundamental, 3, 3)
        K2[:, 0, 0] = focal2
        K2[:, 1, 1] = focal2
        K2[:, 0, 2] = cx2
        K2[:, 1, 2] = cy2

        # get the focal samples
        fov_samples = torch.linspace(
            min_fov, max_fov, num_samples, device=device
        )  # (num_samples,)
        focal_samples = cameras.cx[current_camera_idx] / torch.tan(
            fov_samples * torch.pi / 180.0 / 2.0
        )  # (num_samples,)

        # initialize the votes
        votes = torch.zeros(num_samples, device=device, dtype=dtype)  # (num_samples,)

        # loop over the samples
        for sample_idx, current_focal_sample in enumerate(focal_samples):
            # replace the focal lengths of the current camera
            K1[camera_idx1 == current_camera_idx, 0, 0] = current_focal_sample
            K1[camera_idx1 == current_camera_idx, 1, 1] = current_focal_sample
            K2[camera_idx2 == current_camera_idx, 0, 0] = current_focal_sample
            K2[camera_idx2 == current_camera_idx, 1, 1] = current_focal_sample

            # get sinval ratio for all essential matrices
            essential = (
                K2.transpose(-1, -2) @ fundamental @ K1
            )  # (num_fundamental, 3, 3)
            sinval = torch.linalg.svdvals(essential)  # (num_fundamental, 3)
            sinval_ratio = sinval[:, 0] / (sinval[:, 1] + 1e-12)  # (num_fundamental,)
            assert sinval_ratio.shape == (num_fundamental,)
            del essential, sinval

            # compute the votes
            votes[sample_idx] = torch.exp(-(sinval_ratio - 1.0) / std).sum()

        # compute final fov and focal estimate
        best_fov = fov_samples[votes.argmax()].item()
        best_fov = typing.cast(float, best_fov)
        best_focal = focal_samples[votes.argmax()].item()
        best_focal = typing.cast(float, best_focal)

        # in place update
        cameras.focal[current_camera_idx] = best_focal
        cameras.calibrated = True
        logger.info(
            f"[Camera {current_camera_idx}] Estimated focal : {best_focal} (prior: {cameras.focal_prior[current_camera_idx]})"
        )

        # indicate the camera is estimated
        camera_mask[current_camera_idx] = True
        camera_pair_mask[current_camera_idx] = True
        camera_pair_mask[:, current_camera_idx] = True


@torch.no_grad()
def calibrate_matches(matches: Matches, cameras: Cameras):
    # get input dtype and do computation in float64
    dtype = matches.xy.dtype

    # get intrinsics matrix
    K = cameras.K.to(torch.float64)  # (num_cameras, 3, 3)
    K_inv = torch.inverse(K)  # (num_cameras, 3, 3)

    # calibrate xy
    xy_homo = to_homogeneous(matches.xy)  # (num_images, num_keypoints, 3)
    xy_homo = xy_homo.to(torch.float64)  # (num_images, num_keypoints, 3)
    xy_homo = torch.einsum(
        "nij,nkj->nki", K_inv[cameras.camera_idx], xy_homo
    )  # (num_images, num_keypoints, 3)
    assert torch.all((xy_homo[..., -1] - 1.0).abs()[~xy_homo[..., -1].isnan()] < 1e-6)
    xy_homo = xy_homo.to(dtype)
    matches.xy = xy_homo[..., :-1].contiguous()  # (num_images, num_keypoints, 2)

    # calibrate fundamental
    fundamental = matches.matrix[matches.is_epipolar]  # (num_fundamental, 3, 3)
    fundamental = fundamental.to(torch.float64)  # (num_fundamental, 3, 3)
    essential = (
        K[cameras.camera_idx[matches.image_idx2[matches.is_epipolar]]].transpose(-1, -2)
        @ fundamental
        @ K[cameras.camera_idx[matches.image_idx1[matches.is_epipolar]]]
    )  # (num_fundamental, 3, 3)
    essential = essential.to(dtype)  # (num_fundamental, 3, 3)
    matches.matrix[matches.is_epipolar] = essential

    # calibrate H
    homography = matches.matrix[matches.is_homography]  # (num_homography, 3, 3)
    homography = homography.to(torch.float64)  # (num_homography, 3, 3)
    homography = (
        K_inv[cameras.camera_idx[matches.image_idx2[matches.is_homography]]]
        @ homography
        @ K[cameras.camera_idx[matches.image_idx1[matches.is_homography]]]
    )  # (num_homography, 3, 3)
    homography = homography.to(dtype)  # (num_homography, 3, 3)
    matches.matrix[matches.is_homography] = homography

    # set the calibrated flag
    matches.calibrated = True
