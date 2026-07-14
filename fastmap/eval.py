import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy import spatial
import einops

from fastmap.io import read_model as read_colmap_output


def load_json(fname):
    with open(fname, "r") as f:
        return json.load(f)


def write_json(fname, payload, **kwargs):
    with open(fname, "w") as f:
        json.dump(payload, f, **kwargs)


def angle_error_vec(v1, v2):
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    return np.rad2deg(np.arccos(np.clip(np.dot(v1, v2) / n, -1.0, 1.0)))


def angle_error_mat(R1, R2):
    cos = (np.trace(np.dot(R1.T, R2)) - 1) / 2
    cos = np.round(cos, decimals=4)

    cos = np.clip(cos, -1.0, 1.0)  # numercial errors can make it out of bounds
    return np.rad2deg(np.abs(np.arccos(cos)))


def pose_auc(errors, thresholds, ret_dict=False):
    n = len(errors)
    assert n > 0
    assert (np.array(thresholds) > 0).all()

    errors = np.sort(errors)
    recall = (np.arange(n) + 1) / n

    errors = np.r_[0.0, errors]
    recall = np.r_[0.0, recall]

    aucs = []
    for t in thresholds:
        last_index = np.searchsorted(errors, t)
        # insert (t, recall[last_index - 1]) as the closing tick of the band [0, t]
        # last_index-1 cuz often all pose errors are < t; last_index becomes N; can only use N - 1.

        r = np.r_[recall[:last_index], recall[last_index - 1]]
        e = np.r_[errors[:last_index], t]
        aucs.append(np.trapz(r, x=e) / t)
        # aucs.append(recall[last_index - 1])  RRA

    if ret_dict:
        return {f"auc@{t}": auc for t, auc in zip(thresholds, aucs)}
    else:
        return aucs


def compute_ate(gt, pd):
    '''compute average translation error. used by FlowMap.
    Args:
        gt (torch.Tensor): [N, 3] ground truth translation vectors
        pd (torch.Tensor): [N, 3] predicted translation vectors
    '''
    aligned_gt, aligned_pd, _ = spatial.procrustes(
        gt.detach().cpu().numpy(),
        pd.cpu().numpy(),
    )
    aligned_gt = torch.tensor(aligned_gt, dtype=torch.float32, device=gt.device)
    aligned_pd = torch.tensor(
        aligned_pd, dtype=torch.float32, device=pd.device
    )
    ate = ((aligned_gt - aligned_pd) ** 2).mean() ** 0.5
    return ate.item()


def Rt_to_T(R, t) -> torch.Tensor:
    T = torch.cat([R, t[:, :, None]], dim=-1)  # (B, 3, 4)
    last_row = torch.tensor([0.0, 0.0, 0.0, 1.0], device=T.device, dtype=T.dtype).view(
        1, 1, 4
    )  # (1, 1, 4)
    T = torch.cat([T, last_row.expand(T.shape[0], 1, 4)], dim=1)  # (B, 4, 4)
    return T


def batch_inv_pose(poses):
    # poses: [B, 4, 4]
    Rs = poses[:, 0:3, 0:3]
    ts = poses[:, 0:3, 3]
    n = poses.shape[0]

    T = torch.eye(4, device=poses.device, dtype=poses.dtype).unsqueeze(0).repeat(n, 1, 1)
    T[:, 0:3, 0:3] = einops.rearrange(Rs,  "n a b -> n b a")
    T[:, 0:3, 3] = einops.einsum(Rs, -ts, "n a b, n a -> n b")  # note the R.T
    return T


def _cos_to_angle(cos_theta):
    cos_theta = torch.clip(cos_theta, -1., 1.)
    return torch.rad2deg(
        torch.acos(cos_theta).abs()
    )


def batch_rot_angle_error(R1s, R2s):
    R1s_T = einops.rearrange(R1s,  "n a b -> n b a")
    trace = einops.einsum(torch.bmm(R1s_T, R2s), '... i i -> ...')
    assert torch.allclose(trace, torch.vmap(torch.trace)(torch.bmm(R1s_T, R2s)))
    cos_theta = (trace - 1) / 2
    # WARN: this changes numbers quit a bit when cos_theta is close to 0.
    cos_theta = torch.round(cos_theta, decimals=4)
    return _cos_to_angle(cos_theta)


def batch_vec_angle_error(v1s, v2s):
    v1s = torch.nn.functional.normalize(v1s, dim=-1)
    v2s = torch.nn.functional.normalize(v2s, dim=-1)
    cos_theta = einops.einsum(v1s, v2s, "n d, n d -> n")
    return _cos_to_angle(cos_theta)


@torch.inference_mode()
def pose_pair_angle_error(gt_poses, pd_poses):
    assert gt_poses.shape == pd_poses.shape, f"{gt_poses.shape} vs {pd_poses.shape}"
    n = len(gt_poses)
    inds = torch.combinations(torch.arange(n), 2, with_replacement=False)
    inds = inds.to(gt_poses.device).T  # [2, n_pairs]
    gt_pose_pairs = gt_poses[inds, :, :]
    pd_pose_pairs = pd_poses[inds, :, :]
    del gt_poses, pd_poses

    gt_a2b = torch.bmm(batch_inv_pose(gt_pose_pairs[1]), gt_pose_pairs[0])
    pd_a2b = torch.bmm(batch_inv_pose(pd_pose_pairs[1]), pd_pose_pairs[0])

    r_err = batch_rot_angle_error(gt_a2b[:, :3, :3], pd_a2b[:, :3, :3])
    t_err = batch_vec_angle_error(gt_a2b[:, :3, -1], pd_a2b[:, :3, -1])

    return r_err, t_err


def compute_auc(errs):
    angle_thresholds = [1, 3, 5, 10]
    aucs = pose_auc(errs.cpu().numpy(), angle_thresholds, ret_dict=False)
    aucs = np.array(aucs) * 100
    aucs = aucs.tolist()
    return aucs


def mAA(errs, max_threshold=30):
    """pose diffusion's mAA
    """
    # WARN: rightmost bin edge is 31. Err above 31 degrees are ignored.
    # hence must normalize by total N; np.histogram(density=True) would be incorrect.
    bins = np.arange(max_threshold + 1)

    # Calculate histogram of maximum error values
    histogram, _ = np.histogram(errs, bins=bins)
    # Normalize the histogram
    N = float(len(errs))
    normalized_histogram = histogram.astype(float) / N

    # Compute and return the cumulative sum of the normalized histogram
    return np.mean(np.cumsum(normalized_histogram)).item() * 100


@torch.inference_mode()
def pose_stats_suite(gt_poses, pd_poses):
    stats = {}

    if gt_poses.shape != pd_poses.shape:
        raise ValueError(f"{gt_poses.shape} vs {pd_poses.shape}")

    ate = compute_ate(gt_poses[:, 0:3, 3], pd_poses[:, 0:3, 3])
    stats['ate'] = ate
    del ate

    r_err, t_err = pose_pair_angle_error(gt_poses, pd_poses)
    p_err = torch.max(
        torch.stack([r_err, t_err], dim=0),
        dim=0
    ).values

    stats['auc_p'] = compute_auc(p_err)
    stats['auc_t'] = compute_auc(t_err)
    stats['auc_r'] = compute_auc(r_err)

    def _frac_below(seq, thresh):
        return ((seq < thresh).float().mean() * 100).item()

    stats['RRA'] = [_frac_below(r_err, t) for t in [1, 3, 5, 10, 15]]
    stats['RTA'] = [_frac_below(t_err, t) for t in [1, 3, 5, 10, 15]]
    stats['mAA@30'] = mAA(p_err.cpu().numpy())

    def _frac_above(seq, thresh):
        return ((seq > thresh).float().mean() * 100).item()

    # stats['GRA'] = [_frac_above(r_err, t) for t in [5, 10, 15, 35, 60]]
    # stats['GTA'] = [_frac_above(t_err, t) for t in [5, 10, 20, 30, 60, 90, 120, 150]]

    return stats


def load_colmap_db_cams(fname, ext, return_all=True, device="cuda"):
    colmap_output = read_colmap_output(fname, device=device, ext=ext)
    names = colmap_output.names
    Rs, ts = colmap_output.rotation, colmap_output.translation
    Rs, ts = Rs.to(device), ts.to(device)

    w2c_colmap = Rt_to_T(R=Rs, t=ts)  # [N, 4, 4]
    c2w_colmap = batch_inv_pose(w2c_colmap)

    # assert torch.allclose(c2w_colmap, w2c_colmap.inverse(), atol=1e-4, rtol=1e-2)

    c2w_colmap = c2w_colmap.to(torch.float64)
    return names, c2w_colmap


def load_largest_cam_cluster(res_dir):
    import os
    assert (res_dir / "0").is_dir(), "should have at least one cam cluster."
    cam_clusters = list(sorted(os.listdir(res_dir)))
    N = len(cam_clusters)
    # naming must be 0, 1, 2, etc
    assert cam_clusters == [str(i) for i in range(N)]

    n_cams = 0
    i_max = 0

    if N > 1:
        for i in range(N):
            pd_fnames, pd_poses = load_colmap_db_cams(res_dir / str(i), ".bin")
            n = len(pd_fnames)
            print(f"cluster {i}, n_cams: {n}")
            if n > n_cams:
                n_cams = n
                i_max = i
            del pd_fnames, pd_poses

    pd_fnames, pd_poses = load_colmap_db_cams(res_dir / str(i_max), ".bin")
    return pd_fnames, pd_poses


def read_json_gt_cams(fname):
    cams = load_json(fname)
    fnames = [e['fname'] for e in cams]
    c2w = [e['c2w'] for e in cams]
    c2w = torch.tensor(c2w, dtype=torch.float64, device="cuda")
    return fnames, c2w


def do_eval(pd: Path, gt: Path):
    assert pd.is_dir()
    pd_fnames, pd_poses = load_largest_cam_cluster(Path(pd))

    assert gt.exists()
    if gt.is_dir():
        is_bin, is_txt = (gt / 'cameras.bin').is_file(), (gt / 'cameras.txt').is_file()
        assert is_bin or is_txt
        ext = '.bin' if is_bin else '.txt'
        gt_fnames, gt_poses = load_colmap_db_cams(gt, ext)
    elif gt.is_file():
        gt_fnames, gt_poses = read_json_gt_cams(gt)

    # no duplicate filenames
    assert len(set(gt_fnames)) == len(gt_fnames)
    assert len(set(pd_fnames)) == len(pd_fnames)
    # assert set(pd_fnames).issubset(set(gt_fnames))  # not true for drone deploy
    N = len(gt_fnames)

    # need 1-1 correspondence between pd and gt
    # for missing cameras (due to cam cluster split, failed registration, etc)
    # use 0-th prediction as fill-in. It's not the best way to handle missing cameras, but usable.
    inds = []
    n_fill_in = 0
    for i in range(N):
        if gt_fnames[i] not in pd_fnames:
            n_fill_in += 1
            inds.append(0)
        else:
            # ensures 1-1 mapping for existing cams
            inds.append(pd_fnames.index(gt_fnames[i]))

    if n_fill_in > 0:
        print(f"gt {len(gt_fnames)}; pd {len(pd_fnames)}; fill in {n_fill_in}.")

    inds = torch.tensor(inds).cuda()
    resampled_pd_poses = pd_poses[inds]

    results = pose_stats_suite(gt_poses, resampled_pd_poses)

    out = {}
    out['res'] = results
    out['nums'] = (len(gt_poses), len(pd_poses), n_fill_in)

    write_json("./eval.json", out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pd", type=str, required=True, help="path to prediction directory."
    )
    parser.add_argument(
        "--gt", type=str, required=True, help="path to ground truth json file / colmap directory."
    )

    args = parser.parse_args()
    do_eval(Path(args.pd), Path(args.gt))


if __name__ == "__main__":
    main()
