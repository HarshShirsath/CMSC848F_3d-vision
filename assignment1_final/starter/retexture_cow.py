"""
Sample code to render a cow.

Usage:
    python -m starter.render_mesh --image_size 256 --output_path images/cow_render.jpg
"""
import argparse

import matplotlib.pyplot as plt
import pytorch3d
import torch
import imageio
import numpy as np

from utils import get_device, get_mesh_renderer, load_cow_mesh


def render_cow(
    cow_path="data/cow.obj", image_size=256, color=[0.7, 0.7, 1], device=None,
):
    # The device tells us whether we are rendering with GPU or CPU. The rendering will
    # be *much* faster if you have a CUDA-enabled NVIDIA GPU. However, your code will
    # still run fine on a CPU.
    # The default is to run on CPU, so if you do not have a GPU, you do not need to
    # worry about specifying the device in all of these functions.
    if device is None:
        device = get_device()

    # Get the renderer.
    renderer = get_mesh_renderer(image_size=image_size)

    # Get the vertices, faces, and textures.
    vertices, faces = load_cow_mesh(cow_path)
    vertices = vertices.unsqueeze(0)  # (N_v, 3) -> (1, N_v, 3)
    faces = faces.unsqueeze(0)  # (N_f, 3) -> (1, N_f, 3)

    z_max = vertices[0, :, 2].max()
    z_min= vertices[0, :, 2].min()

    z = vertices[0, :, 2]

    alpha = (z - z_min) / (z_max - z_min)
    # alpha = alpha.view(-1, 1)

    color1 = [0, 0, 1]

    color2 = [1, 0, 0]

    # color = alpha * color2 + (1 - alpha) * color1

    colors = alpha.view(-1, 1) * torch.tensor(color2).view(1, 1, 3) + (1 - alpha.view(-1, 1)) * torch.tensor(color1).view(1, 1, 3)


    textures = torch.ones_like(vertices)  # (1, N_v, 3)
    textures = textures * colors  # (1, N_v, 3)

    mesh = pytorch3d.structures.Meshes(
        verts=vertices,
        faces=faces,
        textures=pytorch3d.renderer.TexturesVertex(textures),
    )
    mesh = mesh.to(device)

    # Prepare the camera:
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(
        R=torch.eye(3).unsqueeze(0), T=torch.tensor([[0, 0, 3]]), fov=60, device=device
    )

    # Place a point light in front of the cow.
    lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)

    output_path = "images/cow_retextured_rotation.gif"
    num_frames = 60
    duration = 15

    # Create views for the 360-degree rotation
    elevations = torch.linspace(0, 360, num_frames, device=device)
    azimuths = torch.linspace(0, 360, num_frames, device=device)
    images = []
    for elevation, azimuth in zip(elevations, azimuths):
        R, T = pytorch3d.renderer.look_at_view_transform(
            dist=3.0,
            elev=elevation,
            azim=azimuth
        )
        cameras.R = R.to(device)
        cameras.T = T.to(device)
        rend = renderer(mesh, cameras=cameras, lights=lights)
        rend = rend[0, ..., :3].cpu().numpy()  # (B, H, W, 4) -> (H, W, 3)
        rend=(rend * 255).astype(np.uint8)
        images.append(rend)

  
    imageio.mimsave(output_path, images, duration=duration)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cow_path", type=str, default="data/cow.obj")
    parser.add_argument("--output_path", type=str, default="images/cow_render.jpg")
    parser.add_argument("--image_size", type=int, default=256)
    args = parser.parse_args()
    # image = render_cow(cow_path=args.cow_path, image_size=args.image_size)
    # plt.imsave(args.output_path, image)

    render_cow(
        cow_path=args.cow_path,
        image_size=args.image_size
    )



