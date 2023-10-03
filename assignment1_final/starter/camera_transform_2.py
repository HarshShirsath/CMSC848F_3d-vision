"""
Usage:
    python -m starter.camera_transforms --image_size 512
"""
import argparse
import os
import matplotlib.pyplot as plt
import pytorch3d
import torch
import numpy as np

from utils import get_device, get_mesh_renderer


def render_cow(
    cow_path="data/cow_with_axis.obj",
    image_size=512,
    R_relative= [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    T_relative=[2, 0, 0],
    device=None,
):
    if device is None:
        device = get_device()
    meshes = pytorch3d.io.load_objs_as_meshes([cow_path]).to(device)

    R_relative = torch.tensor(R_relative).float()
    T_relative = torch.tensor(T_relative).float()

    angle_degrees = -25  # Adjust the rotation angle as needed
    angle_radians = np.radians(angle_degrees)
    rotation_matrix = torch.tensor(
        [
        [np.cos(angle_radians), 0, np.sin(angle_radians)],
        [0, 1, 0],
        [-np.sin(angle_radians), 0, np.cos(angle_radians)],
    ]
    ).float()

    # Define the camera transformation with rotation and translation
    R = rotation_matrix @ R_relative 
    T = R @ torch.tensor([0.0, 0, 3]) + T_relative
    # since the pytorch3d internal uses Point= point@R+t instead of using Point=R @ point+t,
    # we need to add R.t() to compensate that.
    renderer = get_mesh_renderer(image_size=image_size)
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(
        R=R.t().unsqueeze(0), T=T.unsqueeze(0), device=device,
    )
    lights = pytorch3d.renderer.PointLights(location=[[0, 0.0, -3.0]], device=device,)
    rend = renderer(meshes, cameras=cameras, lights=lights)
    return rend[0, ..., :3].cpu().numpy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cow_path", type=str, default="data/cow_with_axis.obj")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--output_path", type=str, default="images/transform 2.jpg")
    args = parser.parse_args()

    plt.imsave(args.output_path, render_cow(cow_path=args.cow_path, image_size=args.image_size))
