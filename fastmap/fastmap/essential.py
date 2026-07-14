import torch


@torch.no_grad()
def decompose_essential(essential: torch.Tensor, dtype: torch.dtype = torch.float64):
    """Compute the relative pose from the essential matrix.

    Args:
        matches: Matches, containing matching points and mask
        dtype: torch.dtype, the data type for the computation

    Returns:
        R: torch.Tensor float (num_epipolar_pairs, 4, 3, 3), 4 solutions of relative rotation (from frame1 to frame2)
        t: torch.Tensor float (num_epipolar_pairs, 4, 3), 4 solutions of relative translation (from frame1 to frame2)
    """
    ########## Prepare ##########
    B = essential.shape[0]

    # switch to computation dtype
    input_dtype = essential.dtype
    E = essential.to(dtype)

    ########## Decompose the fundamental matrix ##########
    # compute the relative pose (all four solutions)
    U, _, Vh = torch.linalg.svd(E)  # (B, 3, 3), (B, 3), (B, 3, 3)
    U[:, :, -1] *= torch.sign(torch.linalg.det(U))[:, None]  # (B, 3, 3)
    Vh[:, -1] *= torch.sign(torch.linalg.det(Vh))[:, None]  # (B, 3, 3)
    W = (
        torch.tensor([[0, -1.0, 0], [1.0, 0, 0], [0, 0, 1.0]])
        .to(E)[None]
        .repeat(B, 1, 1)
    )  # (B, 3, 3)
    Z = (
        torch.tensor([[0, 1.0, 0], [-1.0, 0, 0], [0, 0, 0]]).to(E)[None].repeat(B, 1, 1)
    )  # (B, 3, 3)
    R1 = U @ W @ Vh  # (B, 3, 3)
    R2 = U @ W.transpose(-1, -2) @ Vh  # (B, 3, 3)
    t = U[..., -1]  # (B, 3)
    R = torch.stack([R1, R2, R1, R2], dim=1)  # (B, 4, 3, 3)
    t = torch.stack([t, t, -t, -t], dim=1)  # (B, 4, 3)
    del R1, R2, U, Vh, W, Z

    ########## Return ##########
    # switch back to input dtype
    R = R.to(input_dtype)
    t = t.to(input_dtype)
    return R, t
