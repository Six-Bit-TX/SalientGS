from loguru import logger
import torch

from fastmap.container import Matches, ImagePairs
from fastmap.essential import decompose_essential
from fastmap.homography import decompose_homography
from fastmap.point_pair import cheirality_check, compute_epipolar_error


@torch.no_grad()
def decompose(
    matches: Matches,
    error_thr: float = 0.01,
):
    """Decompose essential and homography matrix to get the relative pose and build the ImagePairs container
    Args:
        matches: Matches, containing matching points and mask
        error_thr: float, epipolar error threshold for disambiguation of 4 solutions
    Returns:
        image_pairs: ImagePairs container
    """
    ##### Decompose essential #####
    assert matches.calibrated
    R_epipolar, t_epipolar = decompose_essential(
        matches.matrix[matches.is_epipolar]
    )  # (num_epipolar_pairs, num_solutions=4, 3, 3), (num_epipolar_pairs, num_solutions=4, 3)

    # make sure there is no nan
    assert not torch.isnan(R_epipolar).any()
    assert not torch.isnan(t_epipolar).any()

    ##### Decompose homography #####
    if torch.any(matches.is_homography):
        R_homography, t_homography, _, valid_mask_homography = decompose_homography(
            matches.matrix[matches.is_homography]
        )  # (num_homography_pairs, num_solutions=4, 3, 3), (num_homography_pairs, num_solutions=4, 3), (num_homography_pairs,)

        # make sure there is no nan
        assert not torch.isnan(R_homography[valid_mask_homography]).any()
        assert not torch.isnan(t_homography[valid_mask_homography]).any()
    else:
        R_homography = None
        t_homography = None
        valid_mask_homography = None

    ##### Merge results #####
    num_image_pairs = len(matches.matrix)
    device = matches.matrix.device
    dtype = matches.matrix.dtype
    R = torch.zeros(
        num_image_pairs, 4, 3, 3, device=device, dtype=dtype
    )  # (num_image_pairs, num_solutions=4, 3, 3)
    t = torch.zeros(
        num_image_pairs, 4, 3, device=device, dtype=dtype
    )  # (num_image_pairs, num_solutions=4, 3)
    valid_mask = torch.ones(
        num_image_pairs, device=device, dtype=torch.bool
    )  # (num_image_pairs,)
    R[matches.is_epipolar] = R_epipolar
    t[matches.is_epipolar] = t_epipolar
    if torch.any(matches.is_homography):
        assert R_homography is not None
        assert t_homography is not None
        assert valid_mask_homography is not None
        R[matches.is_homography] = R_homography
        t[matches.is_homography] = t_homography
        valid_mask[matches.is_homography] = valid_mask_homography
    else:
        assert R_homography is None
        assert t_homography is None
        assert valid_mask_homography is None

    ##### Disambiguate 4 solutions #####
    R, t, num_inliers, ambiguity_mask = disambiguate(
        matches=matches,
        R=R,
        t=t,
        error_thr=error_thr,
    )  # (num_image_pairs, 3, 3), (num_image_pairs, 3), (num_image_pairs,)

    # update valid mask with ambiguity mask
    valid_mask &= ~ambiguity_mask

    ##### Return #####
    image_pairs = ImagePairs(
        image_idx1=matches.image_idx1[valid_mask],
        image_idx2=matches.image_idx2[valid_mask],
        rotation=R[valid_mask],
        translation=t[valid_mask],
        num_inliers=num_inliers[valid_mask],
    )
    logger.info(
        f"After pose decomposition, filter out {(~valid_mask).long().sum().item()} invalid image pairs from {len(valid_mask)} image pairs."
    )
    return image_pairs


@torch.no_grad()
def disambiguate(matches: Matches, R: torch.Tensor, t: torch.Tensor, error_thr: float):
    """Disambiguate 4 solutions of relative pose
    Args:
        matches: Matches, containing matching points and mask
        R: torch.Tensor, (num_image_pairs, 4, 3, 3), relative rotation matrix
        t: torch.Tensor, (num_image_pairs, 4, 3), relative translation vector
        error_thr: float, epipolar error threshold
    Returns:
        R: torch.Tensor, (num_image_pairs, 3, 3), relative rotation matrix
        t: torch.Tensor, (num_image_pairs, 3), relative translation vector
        num_inliers: torch.Tensor, (num_image_pairs,), number of inliers
        ambiguity_mask: torch.Tensor, (num_image_pairs, 4), mask of ambiguity; True if the solution is ambiguous
    """
    # avoid magic numbers
    num_solutions = 4

    # initialize the number of inliers
    num_inliers = torch.zeros(
        matches.num_image_pairs, num_solutions, device=matches.device, dtype=torch.long
    )  # (num_image_pairs, num_solutions)

    # get all the data
    for i, matches_data in enumerate(matches.query()):
        logger.info(f"Disambiuating 4 solutions: batch {i}...")

        # get data
        batch_R = R[matches_data.image_pair_idx]  # (B, num_solutions=4, 3, 3)
        batch_t = t[matches_data.image_pair_idx]  # (B, num_solutions=4, 3)
        batch_xy1 = matches_data.xy1[:, None].expand(
            -1, num_solutions, -1
        )  # (B, num_solutions=4, 2)
        batch_xy2 = matches_data.xy2[:, None].expand(
            -1, num_solutions, -1
        )  # (B, num_solutions=4, 2)

        # cheirality check
        batch_inlier_mask = cheirality_check(
            xy1=batch_xy1,
            xy2=batch_xy2,
            R=batch_R,
            t=batch_t,
        )  # (B, num_solutions=4)

        # epipolar error check
        batch_inlier_mask &= (
            compute_epipolar_error(
                xy1=batch_xy1,
                xy2=batch_xy2,
                R=batch_R,
                t=batch_t,
            )
            < error_thr
        )  # (B, num_solutions=4)

        # avoid misuse
        del batch_R, batch_t

        # accumulate
        index = matches_data.image_pair_idx[:, None].expand(
            -1, num_solutions
        )  # (B, num_solutions)
        src = batch_inlier_mask.long()  # (num_point_pairs, num_solutions)
        num_inliers.scatter_reduce_(
            dim=0, index=index, src=src, reduce="sum"
        )  # (num_image_pairs, num_solutions)
        del index, src

    # find the top2 solutions
    top2 = num_inliers.long().topk(
        k=2, dim=-1
    )  # Tuple[Tensor, Tensor], top2.indices, top2.values

    # get the best solution
    best_idx = top2.indices[:, 0]  # (num_image_pairs,)
    num_inliers = num_inliers[
        torch.arange(len(num_inliers)).to(matches.device), best_idx
    ]  # (num_image_pairs,)
    R = R[torch.arange(len(R)).to(R.device), best_idx]  # (num_image_pairs, 3, 3)
    t = t[torch.arange(len(t)).to(t.device), best_idx]  # (num_image_pairs, 3)

    # get the ambiguity mask
    ambiguity_mask = top2.values[:, 0] == top2.values[:, 1]
    ambiguity_mask = torch.zeros_like(ambiguity_mask)

    # return
    return R, t, num_inliers, ambiguity_mask
