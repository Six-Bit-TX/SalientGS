from loguru import logger
import torch

from fastmap.container import Tracks, Matches


MAX_INT = int(2**31 - 1)


@torch.no_grad()
def _propagate(
    matches: Matches,
    min_track_size: int = 2,
):
    """Propagate track idx to find all the tracks.
    Args:
        matches: Matches container
        min_track_size: int, minimum size of a track
    Returns:
        track_size: torch.tensor, long, shape=(num_valid_tracks,), size of each track
        point_track_idx: torch.tensor, long, shape=(num_images, num_keypoints_per_image), track idx for each point
        point_valid_mask: torch.tensor, bool, shape=(num_images, num_keypoints_per_image), whether the point is in a valid track
    """
    device = matches.device
    num_images = matches.num_images
    num_keypoints_per_image = matches.num_keypoints_per_image

    ########## Propagate the track idx ##########
    # this is the minimum point idx in the same track as the current point, and it will be updated iteratively
    min_global_idx = torch.arange(
        num_images * num_keypoints_per_image, device=device, dtype=torch.long
    )  # (num_images * num_keypoints_per_image,)
    logger.info("Building tracks by iteratively propagating the track idx...")
    for epoch_idx in range(1000000):
        finished = True
        for matches_data in matches.query():
            batch_global_keypoint_idx1 = (
                matches_data.keypoint_idx1
                + matches_data.image_idx1 * matches.num_keypoints_per_image
            )  # (B,)
            batch_global_keypoint_idx2 = (
                matches_data.keypoint_idx2
                + matches_data.image_idx2 * matches.num_keypoints_per_image
            )  # (B,)
            batch_min_idx = torch.min(
                min_global_idx[batch_global_keypoint_idx1],
                min_global_idx[batch_global_keypoint_idx2],
            )
            min_global_idx.scatter_reduce_(
                dim=0,
                index=batch_global_keypoint_idx1,
                src=batch_min_idx,
                reduce="amin",
                include_self=True,
            )
            min_global_idx.scatter_reduce_(
                dim=0,
                index=batch_global_keypoint_idx2,
                src=batch_min_idx,
                reduce="amin",
                include_self=True,
            )
            if torch.any(
                min_global_idx[batch_global_keypoint_idx1]
                != min_global_idx[batch_global_keypoint_idx2]
            ):
                finished = False

        if finished:
            logger.info(f"Track building converged after {epoch_idx + 1} epochs.")
            break

    ########## Find track idx ##########
    _, point_track_idx, track_size = torch.unique(
        min_global_idx, return_inverse=True, return_counts=True, sorted=True
    )  # (num_images * num_keypoints_per_image,), (num_tracks,)

    # reshape track_idx to (num_images, num_keypoints_per_image)
    point_track_idx = point_track_idx.reshape(
        num_images, num_keypoints_per_image
    )  # (num_images, num_keypoints_per_image)

    ########## Initialize a mask to indicate if a track is valid ##########
    valid_track_mask = torch.ones_like(track_size, dtype=torch.bool)  # (num_tracks,)

    ########## Filter out a track if it has too few members ##########
    valid_track_mask &= track_size >= min_track_size  # (num_tracks,)

    ########## Filter out a track if it has two different points from the same image ##########
    # loop over images
    for i in range(num_images):
        unique_track_idx, _, count = torch.unique(
            point_track_idx[i], return_inverse=True, return_counts=True
        )  # (num_unique_track_idx,), (num_keypoints_per_image,), (num_unique_track_idx,)
        valid_track_mask[unique_track_idx[count != 1]] = False

    ########## Re-index tracks to remove invalid tracks ##########
    point_valid_mask = valid_track_mask[
        point_track_idx
    ]  # (num_images, num_keypoints_per_image)
    old2new_track_idx = (
        torch.cumsum(valid_track_mask.long(), dim=0) - 1
    )  # (num_tracks,)
    point_track_idx = old2new_track_idx[
        point_track_idx
    ]  # (num_images, num_keypoints_per_image)
    point_track_idx[~point_valid_mask] = MAX_INT
    track_size = track_size[valid_track_mask]  # (num_valid_tracks,)

    ########## Assert ##########
    assert track_size.dtype == torch.long
    assert point_track_idx.dtype == torch.long
    assert point_valid_mask.dtype == torch.bool
    assert point_track_idx.shape == (num_images, num_keypoints_per_image)
    assert point_valid_mask.shape == (num_images, num_keypoints_per_image)

    ########## Return ##########
    return track_size, point_track_idx, point_valid_mask


@torch.no_grad()
def build_tracks(matches: Matches, min_track_size: int = 2) -> Tracks:
    """Build the tracks container from the matches.
    Args:
        matches: Matches, the matches container
        min_track_size: int, minimum size of a track
    Returns:
        tracks: Tracks, the tracks container
    """
    # build the tracks
    track_size, point_track_idx, point_valid_mask = _propagate(
        matches=matches,
        min_track_size=min_track_size,
    )  # (num_tracks,), (num_images, num_keypoints_per_image), (num_images, num_keypoints_per_image)

    # get point data after removing invalid points
    point_track_idx = point_track_idx[point_valid_mask]  # (num_valid_points,)
    xy_pixels = matches.xy_pixels[point_valid_mask]  # (num_valid_points, 2)
    xy = matches.xy[point_valid_mask]  # (num_valid_points, 2)
    image_idx = (
        torch.arange(matches.num_images, device=matches.device, dtype=torch.long)[
            :, None
        ]
        .expand(matches.num_images, matches.num_keypoints_per_image)
        .clone()[point_valid_mask]
    )  # (num_valid_points,)
    keypoint_idx = (
        torch.arange(
            matches.num_keypoints_per_image, device=matches.device, dtype=torch.long
        )[None, :]
        .expand(matches.num_images, matches.num_keypoints_per_image)
        .clone()[point_valid_mask]
    )  # (num_valid_points,)
    del point_valid_mask

    # re-order the points so that points in the same track are contiguous
    sorted_idx = torch.argsort(point_track_idx)  # (num_valid_points,)
    point_track_idx = point_track_idx[sorted_idx]  # (num_valid_points,)
    xy_pixels = xy_pixels[sorted_idx]  # (num_valid_points, 2)
    xy = xy[sorted_idx]  # (num_valid_points, 2)
    image_idx = image_idx[sorted_idx]  # (num_valid_points,)
    keypoint_idx = keypoint_idx[sorted_idx]  # (num_valid_points,)
    del sorted_idx

    # get the first point idx for each track
    track_start = torch.cat(
        [
            torch.tensor([0], dtype=track_size.dtype, device=track_size.device),
            torch.cumsum(track_size, dim=0)[:-1],
        ],
        dim=0,
    )  # (num_tracks,)

    # build the container
    tracks = Tracks(
        track_size=track_size,
        track_start=track_start,
        track_idx=point_track_idx,
        xy_pixels=xy_pixels,
        xy=xy,
        image_idx=image_idx,
        keypoint_idx=keypoint_idx,
    )

    return tracks
