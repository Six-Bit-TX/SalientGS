import os
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

    result_path = os.path.join(gs_args.data_path, 'gsplat')

    ba_loss_flag = '' if gs_args.ba_loss else '--no-ba_loss '
    pose_opt_flag = '' if gs_args.pose_opt else '--no-pose_opt '

    gs_cmd = (
        f'python {_GSPLAT_JOINT} mcmc_importance '
        f'--data_dir {gs_args.data_path} '
        f'--image_folder_name {image_folder_name} '
        f'--data_factor {data_factor} '
        f'--result_dir {result_path} '
        f'--disable_viewer '
        f'--no-use_wandb '
        f'--strategy.cap-max {cap_max} '
        f'{ba_loss_flag}'
        f'{pose_opt_flag}'
    )

    if gs_args.wandb_name:
        gs_cmd = gs_cmd.replace('--no-use_wandb ', '')
        gs_cmd += f' --use_wandb --wandb_name "{gs_args.wandb_name}"'

    subprocess.run(gs_cmd, shell=True)


def entrypoint():
    run_gaussian_splatting()


if __name__ == '__main__':
    entrypoint()
