from loguru import logger
from typing import List, Union, Tuple
import numpy as np
import torch
import torch.nn.functional as F
import trimesh
import pyrender
import tkinter

from fastmap.io import read_model
from fastmap.container import ColmapModel
from fastmap.utils import vector_to_skew_symmetric_matrix, geometric_median


def build_parallelepiped_mesh(
    base1: Union[List[List], np.ndarray],
    base2: Union[List[List], np.ndarray],
    color: None | Tuple[int, int, int] = None,
):
    """Make a trimesh object of a parallelepiped from two squares of the same size
    Args:
        base1: vertices of the first square in order represented as [[x1, y1, z1], [x2, y2, z2], [x3, y3, z3], [x4, y4, z4]]
        base2: vertices of the second square in order represented as [[x1, y1, z1], [x2, y2, z2], [x3, y3, z3], [x4, y4, z4]]
        color: color of the parallelepiped (optional) represented as uint8 (R, G, B) tuple
    Returns:
        trimesh object of the parallelepiped
    """
    # convert to np array
    if not isinstance(base1, np.ndarray):
        base1 = np.array(base1)
    if not isinstance(base2, np.ndarray):
        base2 = np.array(base2)

    # concat to get all vertices
    vertices = np.vstack([base1, base2])
    del base1, base2

    # create faces: each quad becomes two triangles
    faces = []

    # bottom face (0,1,2,3)
    faces += [[0, 1, 2], [0, 2, 3]]

    # top face (4,5,6,7)
    faces += [[4, 6, 5], [4, 7, 6]]

    # side faces (connect corresponding edges)
    for i in range(4):
        j = (i + 1) % 4
        # indices of lower and upper vertices
        a, b = i, j
        c, d = i + 4, j + 4
        # each side quad split into two triangles
        faces += [[a, b, d], [a, d, c]]

    # convert to np array
    faces = np.array(faces)

    # convert color to uint8 np array
    face_color = np.array([color] * len(faces), dtype=np.uint8) if color else None

    # create mesh
    parallelepiped = trimesh.Trimesh(
        vertices=vertices, faces=faces, face_colors=face_color
    )

    # fix normals
    parallelepiped.fix_normals()  # Ensure normals are consistent

    # return
    return parallelepiped


def build_camera_mesh(
    width: float = 1.0,
    height: float = 0.8,
    focal: float = 1.0,
    thickness: float = 0.05,
    cam_size: float = 1.0,
    edge_color: Tuple[int, int, int] = (120, 0, 0),
    image_plane_color: Tuple[int, int, int] = (255, 0, 0),
    transparent: bool = False,
):
    """Make a mesh of a camera with an image plane and edges. The camera center is at (0, 0, 0) and the image plane is of size [-width/2, height/2] x [-width/2, height/2] at z = focal.
    Args:
        width: float, width of the image plane
        height: float, height of the image plane
        focal: float, focal length of the camera
        thickness: float, thickness of the edges of the camera parallelepiped
        cam_size: float, multiplier for the size of the camera
        edge_color: uint8 (R, G, B) tuple, color for edges
        image_plane_color: uint8 (R, G, B) tuple, color for the image plane
        transparent: bool to make the image plane transparent
    Returns:
        trimesh object of the camera parallelepiped
    """
    # multiply by cam_size
    width *= cam_size
    height *= cam_size
    focal *= cam_size
    thickness *= cam_size
    del cam_size

    # make sure thickness makes sense
    assert focal > thickness / 2.0
    assert width > thickness
    assert height > thickness

    # set image plane limits
    x_min, x_max = -width / 2.0, width / 2.0
    y_min, y_max = -height / 2.0, height / 2.0

    # create the borders of the image plane
    border_top = build_parallelepiped_mesh(
        base1=[
            [x_min - thickness / 2.0, y_min - thickness / 2.0, focal + thickness / 2.0],
            [x_max + thickness / 2.0, y_min - thickness / 2.0, focal + thickness / 2.0],
            [x_max + thickness / 2.0, y_min + thickness / 2.0, focal + thickness / 2.0],
            [x_min - thickness / 2.0, y_min + thickness / 2.0, focal + thickness / 2.0],
        ],
        base2=[
            [x_min - thickness / 2.0, y_min - thickness / 2.0, focal - thickness / 2.0],
            [x_max + thickness / 2.0, y_min - thickness / 2.0, focal - thickness / 2.0],
            [x_max + thickness / 2.0, y_min + thickness / 2.0, focal - thickness / 2.0],
            [x_min - thickness / 2.0, y_min + thickness / 2.0, focal - thickness / 2.0],
        ],
        color=edge_color,
    )
    border_bottom = build_parallelepiped_mesh(
        base1=[
            [x_min - thickness / 2.0, y_max - thickness / 2.0, focal + thickness / 2.0],
            [x_max + thickness / 2.0, y_max - thickness / 2.0, focal + thickness / 2.0],
            [x_max + thickness / 2.0, y_max + thickness / 2.0, focal + thickness / 2.0],
            [x_min - thickness / 2.0, y_max + thickness / 2.0, focal + thickness / 2.0],
        ],
        base2=[
            [x_min - thickness / 2.0, y_max - thickness / 2.0, focal - thickness / 2.0],
            [x_max + thickness / 2.0, y_max - thickness / 2.0, focal - thickness / 2.0],
            [x_max + thickness / 2.0, y_max + thickness / 2.0, focal - thickness / 2.0],
            [x_min - thickness / 2.0, y_max + thickness / 2.0, focal - thickness / 2.0],
        ],
        color=edge_color,
    )
    border_left = build_parallelepiped_mesh(
        base1=[
            [x_min - thickness / 2.0, y_min - thickness / 2.0, focal + thickness / 2.0],
            [x_min + thickness / 2.0, y_min - thickness / 2.0, focal + thickness / 2.0],
            [x_min + thickness / 2.0, y_max + thickness / 2.0, focal + thickness / 2.0],
            [x_min - thickness / 2.0, y_max + thickness / 2.0, focal + thickness / 2.0],
        ],
        base2=[
            [x_min - thickness / 2.0, y_min - thickness / 2.0, focal - thickness / 2.0],
            [x_min + thickness / 2.0, y_min - thickness / 2.0, focal - thickness / 2.0],
            [x_min + thickness / 2.0, y_max + thickness / 2.0, focal - thickness / 2.0],
            [x_min - thickness / 2.0, y_max + thickness / 2.0, focal - thickness / 2.0],
        ],
        color=edge_color,
    )
    border_right = build_parallelepiped_mesh(
        base1=[
            [x_max - thickness / 2.0, y_min - thickness / 2.0, focal + thickness / 2.0],
            [x_max + thickness / 2.0, y_min - thickness / 2.0, focal + thickness / 2.0],
            [x_max + thickness / 2.0, y_max + thickness / 2.0, focal + thickness / 2.0],
            [x_max - thickness / 2.0, y_max + thickness / 2.0, focal + thickness / 2.0],
        ],
        base2=[
            [x_max - thickness / 2.0, y_min - thickness / 2.0, focal - thickness / 2.0],
            [x_max + thickness / 2.0, y_min - thickness / 2.0, focal - thickness / 2.0],
            [x_max + thickness / 2.0, y_max + thickness / 2.0, focal - thickness / 2.0],
            [x_max - thickness / 2.0, y_max + thickness / 2.0, focal - thickness / 2.0],
        ],
        color=edge_color,
    )

    # create the mesh for the image plane
    if transparent:
        image_plane = None
    else:
        image_plane = build_parallelepiped_mesh(
            base1=[
                [
                    x_min + thickness / 2.0,
                    y_min + thickness / 2.0,
                    focal + thickness / 2.0,
                ],
                [
                    x_max - thickness / 2.0,
                    y_min + thickness / 2.0,
                    focal + thickness / 2.0,
                ],
                [
                    x_max - thickness / 2.0,
                    y_max - thickness / 2.0,
                    focal + thickness / 2.0,
                ],
                [
                    x_min + thickness / 2.0,
                    y_max - thickness / 2.0,
                    focal + thickness / 2.0,
                ],
            ],
            base2=[
                [
                    x_min + thickness / 2.0,
                    y_min + thickness / 2.0,
                    focal - thickness / 2.0,
                ],
                [
                    x_max - thickness / 2.0,
                    y_min + thickness / 2.0,
                    focal - thickness / 2.0,
                ],
                [
                    x_max - thickness / 2.0,
                    y_max - thickness / 2.0,
                    focal - thickness / 2.0,
                ],
                [
                    x_min + thickness / 2.0,
                    y_max - thickness / 2.0,
                    focal - thickness / 2.0,
                ],
            ],
            color=image_plane_color,
        )

    # create the meshes for the edges
    top = np.array(
        [
            [-thickness / 2.0, -thickness / 2.0, 0.0],
            [-thickness / 2.0, thickness / 2.0, 0.0],
            [thickness / 2.0, thickness / 2.0, 0.0],
            [thickness / 2.0, -thickness / 2.0, 0.0],
        ]
    )
    edge1 = build_parallelepiped_mesh(
        base1=top,
        base2=top
        + np.array(
            [
                [
                    x_min,
                    y_min,
                    focal - thickness / 2.0,
                ],
            ]
        ),
        color=edge_color,
    )
    edge2 = build_parallelepiped_mesh(
        base1=top,
        base2=top
        + np.array(
            [
                [
                    x_min,
                    y_max,
                    focal - thickness / 2.0,
                ],
            ]
        ),
        color=edge_color,
    )
    edge3 = build_parallelepiped_mesh(
        base1=top,
        base2=top
        + np.array(
            [
                [
                    x_max,
                    y_max,
                    focal - thickness / 2.0,
                ],
            ]
        ),
        color=edge_color,
    )
    edge4 = build_parallelepiped_mesh(
        base1=top,
        base2=top
        + np.array(
            [
                [
                    x_max,
                    y_min,
                    focal - thickness / 2.0,
                ],
            ]
        ),
        color=edge_color,
    )

    # concat all meshes for the camera parallelepiped
    mesh = trimesh.util.concatenate(
        [
            border_top,
            border_bottom,
            border_left,
            border_right,
            edge1,
            edge2,
            edge3,
            edge4,
        ]
    )
    if image_plane is not None:
        mesh = trimesh.util.concatenate([mesh, image_plane])

    # return
    return mesh


@torch.no_grad()
def get_viewer_camera_c2w(
    center: torch.Tensor | Tuple[float, float, float],
    elevation_deg: float,
    azimuth_deg: float,
    distance: float,
):
    """
    Generate a 4x4 camera-to-world matrix (OpenGL convention) given lookat center,
    elevation angle, azimuth angle (both in degrees), and distance.

    Args:
        center (torch.Tensor | Tuple[float, float, float]): (3,) the lookat point
        elevation_deg (float): elevation in degrees from XZ plane towards +Y
        azimuth_deg (float): azimuth in degrees, 0 along -Z, increasing counter-clockwise
        distance (float): distance from center to camera

    Returns:
        torch.Tensor: (4, 4) camera-to-world matrix
    """
    dtype = torch.float32
    if not isinstance(center, torch.Tensor):
        center = torch.tensor(center, dtype=dtype)  # (3,)
    device = center.device
    elevation_rad = torch.deg2rad(torch.tensor(elevation_deg, device=device))
    azimuth_rad = torch.deg2rad(torch.tensor(azimuth_deg, device=device))

    # Camera offset in spherical coordinates
    cam_offset = torch.stack(
        [
            torch.sin(azimuth_rad) * torch.cos(elevation_rad),
            torch.sin(elevation_rad),
            -torch.cos(azimuth_rad) * torch.cos(elevation_rad),
        ]
    )
    cam_pos = center + distance * cam_offset

    # Camera coordinate frame
    forward = center - cam_pos
    forward = forward / torch.norm(forward)

    world_up = torch.tensor([0, 1, 0], device=device, dtype=dtype)
    right = torch.cross(forward, world_up, dim=0)
    right = right / torch.norm(right)

    up = torch.cross(right, forward, dim=0)
    up = up / torch.norm(up)

    # Build c2w matrix
    c2w = torch.eye(4, device=device)
    c2w[:3, 0] = right
    c2w[:3, 1] = up
    c2w[:3, 2] = -forward  # OpenGL convention: camera looks along -Z
    c2w[:3, 3] = cam_pos

    return c2w


@torch.no_grad()
def guess_up_direction(c2w):
    """
    Guess the global up direction given a set of OpenCV-style c2w matrices.

    Args:
        c2w (torch.Tensor): shape (N, 4, 4), camera-to-world matrices

    Returns:
        torch.Tensor: (3,), normalized up direction in world coordinates
    """
    assert c2w.ndim == 3 and c2w.shape[1:] == (4, 4), "Input must be (N, 4, 4)"

    # Extract the -Y direction for each camera
    # Camera axes:
    # X: right (c2w[:, :, 0])
    # Y: down (c2w[:, :, 1]) --> so -Y is -c2w[:, :, 1]
    # Z: forward (c2w[:, :, 2])
    camera_neg_y = -c2w[:, :3, 1]  # shape (N, 3)

    # Average the directions
    up = camera_neg_y.mean(dim=0)
    # Normalize
    up = up / torch.norm(up)
    # check
    assert up.shape == (3,)

    return up


@torch.no_grad()
def rotation_matrix_from_axis_angle(
    axis: torch.Tensor | Tuple[float, float, float], angle: float | torch.Tensor
):
    """
    Compute 3x3 rotation matrix from axis and angle.

    Args:
        axis (torch.Tensor | Tuple[float, float, float]): (3,) rotation axis (should be normalized)
        angle (float or torch scalar): rotation angle in radians

    Returns:
        torch.Tensor: (3, 3) rotation matrix
    """
    if not isinstance(axis, torch.Tensor):
        axis = torch.tensor(axis, dtype=torch.float32)  # (3,)
    device = axis.device
    dtype = axis.dtype
    if not isinstance(angle, torch.Tensor):
        angle = torch.tensor(angle, device=device, dtype=dtype)  # (,)
    axis = axis / torch.norm(axis)  # (3,)

    K = vector_to_skew_symmetric_matrix(axis)  # (3, 3)
    I = torch.eye(3, device=device)  # (3, 3)

    R = I + torch.sin(angle) * K + (1 - torch.cos(angle)) * (K @ K)
    return R


@torch.no_grad()
def align_up(
    c2w: torch.Tensor,
    xyz: torch.Tensor,
    up_vector: torch.Tensor | Tuple[float, float, float],
    target_up: torch.Tensor | Tuple[float, float, float] = (0.0, 1.0, 0.0),
):
    """
    Align the up direction of a set of cameras and points to a target up direction.

    Args:
        c2w (torch.Tensor): (num_cameras, 4, 4) camera-to-world matrix
        xyz (torch.Tensor): (num_points, 3) point in world coordinates
        up_vector (torch.Tensor): (3,) estimated up direction (should be normalized)
        target_up (torch.Tensor): (3,) desired up direction (default [0, 1, 0])

    Returns:
        c2w (torch.Tensor): (num_cameras, 4, 4) aligned camera-to-world matrix
        xyz (torch.Tensor): (num_points, 3) aligned points in world coordinates
    """
    # pre-process inputs
    c2w = c2w.clone()  # (num_cameras, 4, 4)
    xyz = xyz.clone()  # (num_points, 3)
    device = c2w.device
    dtype = c2w.dtype
    if not isinstance(up_vector, torch.Tensor):
        up_vector = torch.tensor(up_vector, device=device, dtype=dtype)  # (3,)
    if not isinstance(target_up, torch.Tensor):
        target_up = torch.tensor(target_up, device=device, dtype=dtype)  # (3,)
    up_vector = up_vector / torch.norm(up_vector)  # (3,)
    target_up = target_up / torch.norm(target_up)  # (3,)
    assert isinstance(up_vector, torch.Tensor) and up_vector.shape == (3,)
    assert isinstance(target_up, torch.Tensor) and target_up.shape == (3,)

    axis = torch.cross(up_vector, target_up, dim=0)  # (3,)
    axis = F.normalize(axis, dim=0)  # (3,)
    angle = torch.acos(torch.clamp(torch.dot(up_vector, target_up), -1.0, 1.0))  # (,)

    R = rotation_matrix_from_axis_angle(axis=axis, angle=angle)  # (3, 3)

    c2w[:, :3, :3] = torch.einsum("ij,njk->nik", R, c2w[:, :3, :3])
    c2w[:, :3, 3] = torch.einsum("ij,nj->ni", R, c2w[:, :3, 3])
    xyz = torch.einsum("ij,nj->ni", R, xyz)  # (num_points, 3)

    return c2w, xyz


@torch.no_grad()
def visualize(
    R_w2c: torch.Tensor,
    t_w2c: torch.Tensor,
    xyz: torch.Tensor,
    rgb: torch.Tensor,
    cam_size: float = 0.1,
    point_size: None | float = None,
    elevation_deg: float = 45.0,
    distance: float = 3.0,
    rotate: bool = False,
    rotate_rate: float = 30.0,
    viewport_factor: float = 0.75,
):
    """Run a on-screen viewer to visualize the camera poses and point cloud.
    Args:
        R_w2c: torch.Tensor, float, (num_images, 3, 3) rotation matrices from world to camera coordinates
        t_w2c: torch.Tensor, float, (num_images, 3) translation vectors from world to camera coordinates
        xyz: torch.Tensor, float, (num_points, 3) point cloud in world coordinates
        rgb: torch.Tensor, uint8, (num_points, 3) color of the point cloud
        cam_size: float, size of the cameras (the unit is meaningless, you should tune it by trial and error)
        point_size: float, size of the points in pixels. If None, it will be automatically determined according to the viewport size
        elevation_deg: float, elevation angle of the viewing camera in degrees
        distance: float, distance of the viewing camera from the origin
        rotate: bool, whether to automatically rotate the viewing camera
        rotate_rate: float, rotation speed in degrees per second
        viewport_factor: float, final size of the viewport will be the screen resolution multiplied by this factor
    """
    # get information
    device, dtype = R_w2c.device, R_w2c.dtype
    num_images = R_w2c.shape[0]
    num_points = xyz.shape[0]
    assert R_w2c.shape == (num_images, 3, 3)
    assert t_w2c.shape == (num_images, 3)
    assert xyz.shape == (num_points, 3)
    assert rgb.shape == (num_points, 3)
    assert R_w2c.dtype == t_w2c.dtype == xyz.dtype
    assert rgb.dtype == torch.uint8

    # get 4x4 camera-to-world matrices
    c2w = torch.zeros(
        num_images, 4, 4, device=device, dtype=dtype
    )  # (num_images, 4, 4)
    c2w[:, :3, :3] = R_w2c.transpose(-1, -2)
    c2w[:, :3, 3] = torch.einsum("nij,nj->ni", R_w2c.transpose(-1, -2), -t_w2c)
    c2w[:, 3, 3] = 1.0
    del R_w2c, t_w2c

    # clone everything for safety
    c2w = c2w.clone()  # (num_images, 4, 4)
    xyz = xyz.clone()  # (num_points, 3)
    rgb = rgb.clone()  # (num_points, 3)

    # re-center and normalize everything
    logger.info("Re-centering and normalizing the scene...")
    c2w_center = geometric_median(c2w[:, :3, 3])  # (3,)
    assert c2w_center.shape == (3,)
    c2w_scale = geometric_median(
        (c2w[:, :3, 3] - c2w_center[None, :]).norm(dim=-1, keepdim=True)
    ).item()  # (,)
    assert c2w_scale > 0.0, "Scale is zero, cannot normalize"
    c2w[:, :3, 3] -= c2w_center[None, :]  # (num_images, 3)
    c2w[:, :3, 3] /= c2w_scale  # (num_images, 3)
    xyz -= c2w_center[None, :]  # (num_points, 3)
    xyz /= c2w_scale  # (num_points, 3)
    del c2w_center, c2w_scale

    # align the up direction with the world y axis
    logger.info("Aligning the up direction with the world y axis...")
    up_direction = guess_up_direction(c2w)  # (3,)
    c2w, xyz = align_up(
        c2w=c2w,
        xyz=xyz,
        up_vector=up_direction,
        target_up=(0.0, 1.0, 0.0),
    )  # (num_images, 4, 4), (num_points, 3)
    del up_direction

    # build the scene
    logger.info("Building the scene...")
    scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0])

    # add cameras and point cloud to scene
    cameras = pyrender.Mesh.from_trimesh(
        build_camera_mesh(cam_size=cam_size), smooth=False, poses=c2w.cpu().numpy()
    )
    points3d = pyrender.Mesh.from_points(
        xyz.cpu().numpy(), colors=rgb.float().cpu().numpy() / 255.0
    )
    scene.add(cameras)
    scene.add(points3d)
    del c2w, xyz, rgb
    logger.info(f"Added {num_images} cameras and {num_points} points to the scene.")

    # compute viewport size
    viewport_width = int(tkinter.Tk().winfo_screenwidth() * viewport_factor)
    viewport_height = int(tkinter.Tk().winfo_screenheight() * viewport_factor)
    aspect_ratio = viewport_width / viewport_height
    logger.info(f"Using viewport size: {viewport_width} x {viewport_height}.")

    # heuristically compute the point size according to the viewport size
    if point_size is None:
        point_size = max(1.0, viewport_width / 1024.0)
    logger.info(f"Using point size: {point_size:.2f} pixels.")

    # set main viewing camera in the scene
    viewer_cam_c2w = get_viewer_camera_c2w(
        center=(0.0, 0.0, 0.0),
        elevation_deg=elevation_deg,
        azimuth_deg=0.0,
        distance=distance,
    )
    scene.add(
        pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=aspect_ratio),
        pose=viewer_cam_c2w.cpu().numpy(),
    )

    # create viewer
    logger.info("Creating the viewer...")
    viewer = pyrender.Viewer(
        scene,
        point_size=point_size,
        viewport_size=(
            viewport_width,
            viewport_height,
        ),
        run_in_thread=False,
        view_center=(0.0, 0.0, 0.0),
        window_title="FastMap",
        auto_start=False,
        flat=True,  # no shading, just color
        rotate=rotate,
        rotate_axis=(0.0, 1.0, 0.0),  # rotate around Y axis
        rotate_rate=rotate_rate / 180.0 * 3.14,  # rotation speed (radians per second)
    )

    # show
    logger.info("Viewer running...")
    viewer.start()

    # end
    logger.info("Viewer closed.")


if __name__ == "__main__":
    import argparse

    # define action to parse key=value pairs
    class ParseKwargs(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            out_dict = {}
            if values is not None:
                for value in values:
                    # split to key and value strings
                    if "=" not in value:
                        raise argparse.ArgumentTypeError(
                            f"Expected key=value format, got '{value}'"
                        )
                    key, val = value.split("=", 1)
                    key = key.strip()
                    val = val.strip()
                    del value

                    # convert to appropriate type
                    if val.lower() in {"true", "false"}:
                        val = val.lower() == "true"
                    else:
                        try:
                            val = int(val)
                        except ValueError:
                            pass
                        try:
                            val = float(val)
                        except ValueError:
                            pass

                    # add to dictionary
                    out_dict[key] = val

            # set the parsed dictionary to the namespace
            setattr(namespace, self.dest, out_dict)

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to the model directory (e.g. output/dir/sparse/0/)",
    )
    parser.add_argument(
        "--viewer_options",
        nargs="*",
        action=ParseKwargs,
        default={},
        help="See the viewer options in the docstring of the `visualize` function."
        "Usage: --viewer_options key1=value1 key2=value2 ..."
        "Example: --viewer_options rotate=true point_size=1.0",
    )
    args = parser.parse_args()

    # read model
    model: ColmapModel = read_model(args.model_path)
    assert model.points3d is not None, "No points3d found in the model"
    assert model.rgb is not None, "No rgb found in the model"

    # visualize
    visualize(
        R_w2c=model.rotation,
        t_w2c=model.translation,
        xyz=model.points3d,
        rgb=model.rgb,
        **args.viewer_options,
    )
