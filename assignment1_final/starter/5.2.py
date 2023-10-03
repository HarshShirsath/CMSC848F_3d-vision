import argparse
import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import pytorch3d
from utils import get_points_renderer

def render_torus_gif(image_size=512, num_samples=1000, num_frames=60, duration=15, output_path="images/torus_gif_5.2.2.gif"):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    major_radius = 1.0
    minor_radius = 0.3
    phi = torch.linspace(0, 2 * np.pi, num_samples)
    theta = torch.linspace(0, 2 * np.pi, num_samples)
    Phi, Theta = torch.meshgrid(phi, theta)

    x = (major_radius + minor_radius * torch.cos(Theta)) * torch.cos(Phi)
    y = (major_radius + minor_radius * torch.cos(Theta)) * torch.sin(Phi)
    z = minor_radius * torch.sin(Theta)

    points = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=1)
    color = (points - points.min()) / (points.max() - points.min())

    torus_point_cloud = pytorch3d.structures.Pointclouds(
        points=[points], features=[color],
    ).to(device)

    cameras = pytorch3d.renderer.FoVPerspectiveCameras(T=[[0, 0, 3]], device=device)
    renderer = get_points_renderer(image_size=image_size, device=device)

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
        rend = renderer(torus_point_cloud, cameras=cameras)
        rend = (rend[0, ..., :3] * 255).cpu().numpy().astype(np.uint8)
        images.append(rend)

    # Save the images as a GIF
    imageio.mimsave(output_path, images, duration=duration)
    return images

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--num_frames", type=int, default=60)
    parser.add_argument("--duration", type=int, default=15)
    parser.add_argument("--output_path", type=str, default="images/torus_gif_5.2.2.gif")
    args = parser.parse_args()

    images = render_torus_gif(
        image_size=args.image_size,
        num_samples=args.num_samples,
        num_frames=args.num_frames,
        duration=args.duration,
        output_path=args.output_path
    )

    # You can also display the last frame (optional)
    plt.imshow(images[-1])
    plt.show()