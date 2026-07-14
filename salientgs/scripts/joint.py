import os
import sys
from argparse import ArgumentParser
import subprocess

from salientgs.controllers.data_reader import ReadData, PathInfo

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_GSPLAT_JOINT = os.path.normpath(
    os.path.join(_SCRIPT_DIR, os.pardir, "vis", "gsplat_joint.py")
)


def run_gaussian_splatting():
    parser = ArgumentParser()
    parser.add_argument('--data_path', required=True, help='Path to the data folder')
    parser.add_argument('--cap-max', type=int, default=1500000, help='Maximum number of Gaussians')
    parser.add_argument('--data-factor', type=int, default=1, help='Downsample factor for images')
    parser.add_argument('--result-dir', default=None, help='Output directory (default: DATA_PATH/gsplat)')
    parser.add_argument('--max-steps', type=int, default=30000, help='Exact optimization steps')
    parser.add_argument('--seed', type=int, default=42, help='Training random seed')
    parser.add_argument('--no-ba_loss', dest='ba_loss', action='store_false',
                        help='Disable BA loss during training')
    parser.add_argument('--no-pose_opt', dest='pose_opt', action='store_false',
                        help='Disable pose optimization')
    parser.set_defaults(ba_loss=True, pose_opt=True)
    parser.add_argument('--wandb_name', type=str, default=None,
                        help='Wandb run name (default: auto-generated)')
    gs_args = parser.parse_args()

    path_info = ReadData(gs_args.data_path)
    if not path_info:
        print('Invalid data path, please check the provided path')
        return

    image_folder_name = path_info.image_path.split(os.path.sep)[-1]
    cap_max = getattr(gs_args, 'cap_max')
    data_factor = getattr(gs_args, 'data_factor')

    result_path = gs_args.result_dir or os.path.join(gs_args.data_path, 'gsplat')

    gs_cmd = [
        sys.executable, _GSPLAT_JOINT, 'mcmc_importance',
        '--data-dir', gs_args.data_path,
        '--image-folder-name', image_folder_name,
        '--data-factor', str(data_factor),
        '--result-dir', result_path,
        '--max-steps', str(gs_args.max_steps),
        '--seed', str(gs_args.seed),
        '--disable_viewer', '--no-use_wandb',
        '--strategy.cap-max', str(cap_max),
    ]
    if not gs_args.ba_loss:
        gs_cmd.append('--no-ba-loss')
    if not gs_args.pose_opt:
        gs_cmd.append('--no-pose-opt')

    if gs_args.wandb_name:
        gs_cmd.remove('--no-use_wandb')
        gs_cmd.extend(['--use-wandb', '--wandb-name', gs_args.wandb_name])

    subprocess.run(gs_cmd, check=True)


def entrypoint():
    run_gaussian_splatting()


if __name__ == '__main__':
    entrypoint()
