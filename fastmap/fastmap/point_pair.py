from loguru import logger
import typing
import math
import torch
import torch.nn.functional as F

from fastmap.container import PointPairs, Tracks
from fastmap.utils import to_homogeneous


@torch.no_grad()
def point_pairs_from_tracks(tracks: Tracks) -> PointPairs:
    """Get all the possible point pairs implied by the tracks
    Args:
        tracks: Tracks container
    Returns:
        point_pairs: PointPairs container
    """
    ########## Get useful information ##########
    # get the device
    device = tracks.device
    # get the number of tracks
    num_tracks = tracks.num_tracks
    # get the min and max number of points in a track
    min_track_size = tracks.track_size.min().item()
    max_track_size = tracks.track_size.max().item()
    min_track_size = typing.cast(int, min_track_size)
    max_track_size = typing.cast(int, max_track_size)
    # make sure the track idx of points are sorted
    assert torch.all(tracks.track_idx == tracks.track_idx.sort().values)

    ########## Initialize the memory for point pairs ##########
    num_point_pairs = (tracks.track_size * (tracks.track_size - 1) // 2).sum().item()
    num_point_pairs = typing.cast(int, num_point_pairs)
    point_idx1 = 230492843028 + torch.zeros(
        num_point_pairs, dtype=torch.long, device=device
    )  # (num_point_pairs,)
    point_idx2 = 230492843028 + torch.zeros(
        num_point_pairs, dtype=torch.long, device=device
    )  # (num_point_pairs,)

    # this is the current location to write to
    pointer = 0

    ########## Get all the possible point pairs ##########

    # batch tracks with the same size
    for ts in range(min_track_size, max_track_size + 1):
        # get the point idx for tracks with size ts
        point_idx = torch.nonzero(
            tracks.track_size[tracks.track_idx] == ts, as_tuple=True
        )[
            0
        ]  # (num_points_ts,)
        # this reshaping uses the fact that points in the same track are grouped together in the Tracks container
        point_idx = point_idx.view(-1, ts).contiguous()  # (num_tracks_ts, ts)

        # get the number of tracks of size ts
        num_tracks_ts = point_idx.shape[0]
        if num_tracks_ts == 0:
            continue

        # get the combination idx
        comb = torch.combinations(
            torch.arange(ts, device=device), r=2, with_replacement=False
        )  # (num_comb, 2)

        # get all the combinations of point idx
        num_point_pairs_per_track = comb.shape[0]
        point_idx_pair = torch.gather(
            input=point_idx[:, None, :].repeat(1, num_point_pairs_per_track, 1),
            dim=-1,
            index=comb[None, :, :].repeat(num_tracks_ts, 1, 1),
        )  # (num_tracks_ts, num_comb, 2)
        point_idx_pair = point_idx_pair.view(-1, 2)  # (num_point_pairs_ts, 2)

        # write and increment the pointer
        point_idx1[pointer : pointer + len(point_idx_pair)] = point_idx_pair[:, 0]
        point_idx2[pointer : pointer + len(point_idx_pair)] = point_idx_pair[:, 1]
        pointer += len(point_idx_pair)

        logger.info(
            f"Found {point_idx_pair.shape[0]} point pairs for {point_idx.shape[0]} tracks of size {ts}."
        )

    ######### Build the container ##########
    # sanity check
    assert torch.all(tracks.track_idx[point_idx1] == tracks.track_idx[point_idx2])

    # form the container
    point_pairs = PointPairs(
        xy_pixels=tracks.xy_pixels,
        xy=tracks.xy,
        image_idx=tracks.image_idx,
        keypoint_idx=tracks.keypoint_idx,
        track_idx=tracks.track_idx,
        point_idx1=point_idx1,
        point_idx2=point_idx2,
    )

    logger.info(
        f"Found {point_pairs.point_idx1.shape[0]} point pairs in total from {num_tracks} tracks."
    )

    return point_pairs


@torch.no_grad()
def compute_ray_angle(
    xy1: torch.Tensor,
    xy2: torch.Tensor,
    R: torch.Tensor,
):
    """Compute the angle between the rays of two calibrated 2D correspondences.

    Args:
        xy1: torch.Tensor float (..., 2), calibrated 2D coordinates of the first image
        xy2: torch.Tensor float (..., 2), calibrated 2D coordinates of the second image
        R: torch.Tensor float (..., 3, 3), relative rotation (from frame1 to frame2)
        t: torch.Tensor float (..., 3), relative translation (from frame1 to frame2)

    Returns:
        angle: torch.Tensor float (...,), the angle between the rays in degrees
    """
    # get unnormlized rays in second camera frame
    ray1 = torch.einsum("...ij,...j->...i", R, to_homogeneous(xy1))  # (..., 3)
    ray2 = to_homogeneous(xy2)  # (..., 3)

    # normalize the rays
    ray1 = F.normalize(ray1, p=2, dim=-1)  # (..., 3)
    ray2 = F.normalize(ray2, p=2, dim=-1)  # (..., 3)

    # compute the angle
    angle = torch.acos(
        (ray1 * ray2).sum(dim=-1).clamp(-1.0 + 1e-8, 1.0 - 1e-8)
    )  # (...,)
    angle = angle * 180.0 / math.pi  # (...,)

    # return
    return angle


@torch.no_grad()
def cheirality_check(
    xy1: torch.Tensor,
    xy2: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor,
):
    """Check the cheirality of calibrated 2D correspondences. We do not triangulate the points, but instead check the cheirality of the rays. This is much faster than triangulating the points.

    Args:
        xy1: torch.Tensor float (..., 2), calibrated 2D coordinates of the first image
        xy2: torch.Tensor float (..., 2), calibrated 2D coordinates of the second image
        R: torch.Tensor float (..., 3, 3), relative rotation (from frame1 to frame2)
        t: torch.Tensor float (..., 3), relative translation (from frame1 to frame2)

    Returns:
        mask: torch.Tensor bool (...,), True if the point pair has positive depth
    """
    # get unnormlized rays in second camera frame
    ray1 = torch.einsum("...ij,...j->...i", R, to_homogeneous(xy1))  # (..., 3)
    ray2 = to_homogeneous(xy2)  # (..., 3)

    # compute the mask (note that here t is vector from 2 to 1 in camera 2 frame)
    n1 = torch.cross(ray1, t, dim=-1)  # (..., 3)
    n2 = torch.cross(ray2, t, dim=-1)  # (..., 3)
    n3 = torch.cross(ray1, ray2, dim=-1)  # (..., 3)
    mask = (
        ((n1 * n2).sum(dim=-1) > 0.0)
        & ((n1 * n3).sum(dim=-1) > 0.0)
        & ((n2 * n3).sum(dim=-1) > 0.0)
    )  # (...,)

    # return
    return mask


@torch.no_grad()
def compute_epipolar_error(
    xy1: torch.Tensor,
    xy2: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor,
):
    """Compute epipolar error of calibrated 2D correspondences.

    Args:
        xy1: torch.Tensor float (..., 2), calibrated 2D coordinates of the first image
        xy2: torch.Tensor float (..., 2), calibrated 2D coordinates of the
        R: torch.Tensor float (..., 3, 3), relative rotation (from frame1 to frame2)
        t: torch.Tensor float (..., 3), relative translation (from frame1 to frame2)

    Returns:
        error: torch.Tensor float (...,), the epipolar error
    """
    xy_homo1 = F.normalize(to_homogeneous(xy1), p=2, dim=-1)  # (..., 3)
    xy_homo2 = F.normalize(to_homogeneous(xy2), p=2, dim=-1)  # (..., 3)
    Rx1 = torch.einsum(
        "...ij,...j->...i",
        R,
        xy_homo1,
    )  # (..., 3)
    line = torch.cross(t, Rx1, dim=-1)  # (..., 3)
    error = (line * xy_homo2).sum(dim=-1).abs()  # (...,)

    return error
