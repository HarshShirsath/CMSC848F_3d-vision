# import argparse
# import pickle
# import imageio
# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# from pytorch3d.renderer import look_at_view_transform, FoVPerspectiveCameras
# from pytorch3d.structures import Pointclouds
# from utils import unproject_depth_image
# import os

# # Function to load RGB-D data from the pkl file
# def load_rgbd_data(data_file):
#     with open(data_file, "rb") as f:
#         rgbd_data = pickle.load(f)
#     return rgbd_data

# # Function to create a point cloud from RGB-D data
# def create_point_cloud(rgb, depth, mask, camera):
#     h, w = depth.shape
#     pixel_coords = torch.stack(torch.meshgrid([torch.arange(h), torch.arange(w)]), dim=-1).to(depth.device)
#     points3D = unproject_depth_image(pixel_coords, depth, camera)
#     colors = rgb[mask > 0].view(-1, 3) / 255.0
#     return Pointclouds(points=points3D.view(-1, 3), features=colors)

# def main():
#     parser = argparse.ArgumentParser(description="Render and visualize RGB-D point clouds.")
#     parser.add_argument("--data_file", type=str, default="data/rgbd_data.pkl")
#     parser.add_argument("--output_dir", type=str, default="images/rgbd.gif")
#     parser.add_argument("--image_size", type=int, default=512)
#     parser.add_argument("--num_frames", type=int, default=60)
#     parser.add_argument("--duration", type=int, default=15)
#     args = parser.parse_args()

#     # Load RGB-D data
#     rgbd_data = load_rgbd_data(args.data_file)
    
#     # Extract camera parameters from the loaded data
#     camera1_params = rgbd_data["pose1"]  # Adjust the key based on your data structure
#     camera2_params = rgbd_data["pose2"]  # Adjust the key based on your data structure

#     # Create cameras for both poses
#     camera1 = FoVPerspectiveCameras(fov=60, device="cpu", R=camera1_params["R"], T=camera1_params["T"])
#     camera2 = FoVPerspectiveCameras(fov=60, device="cpu", R=camera2_params["R"], T=camera2_params["T"])

#     # ... Rest of the code


#     # Create point clouds for both images
#     pc1 = create_point_cloud(rgbd_data["image1"], rgbd_data["depth1"], rgbd_data["mask1"], camera1)
#     pc2 = create_point_cloud(rgbd_data["image2"], rgbd_data["depth2"], rgbd_data["mask2"], camera2)
    
#     # Combine point clouds
#     combined_pc = Pointclouds(points=torch.cat([pc1.points_packed(), pc2.points_packed()]), 
#                              features=torch.cat([pc1.features_packed(), pc2.features_packed()]))

#     # Define camera parameters for rendering
#     elevations = torch.linspace(0, 360, args.num_frames)
#     azimuths = torch.linspace(0, 360, args.num_frames)

#     # Create output directory
#     os.makedirs(args.output_dir, exist_ok=True)

#     # Render and save GIFs for each point cloud
#     for i, pc in enumerate([pc1, pc2, combined_pc]):
#         images = []
#         for elevation, azimuth in zip(elevations, azimuths):
#             R, T = look_at_view_transform(dist=6.0, elev=elevation, azim=azimuth)
#             cameras = FoVPerspectiveCameras(fov=60, device="cpu", R=R, T=T)
#             rendered_image = pc.renderer(cameras=cameras)
#             rendered_image = (rendered_image[0, ..., :3] * 255).cpu().numpy().astype(np.uint8)
#             images.append(rendered_image)

#         output_path = os.path.join(args.output_dir, f"point_cloud_{i}.gif")
#         imageio.mimsave(output_path, images, duration=args.duration)
#         print(f"Saved GIF to {output_path}")

# if __name__ == "__main__":
#     main()





"""
Sample code to render various representations.

Usage:
    python -m starter.render_generic --render point_cloud  # 5.1
    python -m starter.render_generic --render parametric  --num_samples 100  # 5.2
    python -m starter.render_generic --render implicit  # 5.3
"""
import argparse
import pickle

import matplotlib.pyplot as plt
import mcubes
import numpy as np
import pytorch3d
import torch
import imageio
from PIL import Image, ImageDraw

from utils import get_device, get_mesh_renderer, get_points_renderer, unproject_depth_image


def load_rgbd_data(path="data/rgbd_data.pkl", image_size=512, duration=200,device = None, output_file="images/plant_2.gif", ):
    with open(path, "rb") as f:
        data = pickle.load(f)
    # return data

    if device is None:
        device = get_device()

    # Unproject both depth images into point clouds
    points1, rgba1 = unproject_depth_image(torch.tensor(data['rgb1']), torch.tensor(data['mask1']), torch.tensor(data['depth1']), data['cameras1'])
    point_cloud1 = pytorch3d.structures.Pointclouds(points=[points1], features=[rgba1]).to(device)
    
    points2, rgba2 = unproject_depth_image(torch.tensor(data['rgb2']), torch.tensor(data['mask2']), torch.tensor(data['depth2']), data['cameras2'])
    point_cloud2 = pytorch3d.structures.Pointclouds(points=[points2], features=[rgba2]).to(device)
    # torch.save(point_cloud1, "images/point_cloud1.gif")
    # torch.save(point_cloud2, "images/point_cloud2.gif")

    # Concatenate the point clouds and color values
    points = torch.cat((points1, points2), dim=0)
    rgba = torch.cat((rgba1, rgba2), dim=0)
    point_cloud3 = pytorch3d.structures.Pointclouds(points=[points], features=[rgba]).to(device)

    renders = []
    for theta in range(0, 360, 10):
        R = torch.tensor([
            [np.cos(np.radians(theta)), 0.0, -np.sin(np.radians(theta))],
            [0.0, 1.0, 0.0],
            [np.sin(np.radians(theta)), 0.0, np.cos(np.radians(theta))]
        ])
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R.unsqueeze(0), T=[[0, 0, 6]], device=device)
        renderer = get_points_renderer(image_size=image_size, device=device)
        rend = renderer(point_cloud2, cameras=cameras)
        # return rend[0, ..., :3].cpu().numpy()
        rend = rend[0, ..., :3].cpu().numpy()
        renders.append(rend)

    images = []
    for i, r in enumerate(renders):
        image = Image.fromarray((r * 255).astype(np.uint8))
        draw = ImageDraw.Draw(image)
        # draw.text((20, 20), f"fov: {fovs[i]:.2f}", fill=(255, 0, 0))
        images.append(np.array(image))
    imageio.mimsave(output_file, images, duration=duration)

    return rend


def render_bridge(
    point_cloud_path="data/bridge_pointcloud.npz",
    image_size=256,
    background_color=(1, 1, 1),
    device=None,
):
    """
    Renders a point cloud.
    """
    if device is None:
        device = get_device()
    renderer = get_points_renderer(
        image_size=image_size, background_color=background_color
    )
    point_cloud = np.load(point_cloud_path)
    verts = torch.Tensor(point_cloud["verts"][::50]).to(device).unsqueeze(0)
    rgb = torch.Tensor(point_cloud["rgb"][::50]).to(device).unsqueeze(0)
    point_cloud = pytorch3d.structures.Pointclouds(points=verts, features=rgb)
    R, T = pytorch3d.renderer.look_at_view_transform(4, 10, 0)
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
    rend = renderer(point_cloud, cameras=cameras)
    rend = rend.cpu().numpy()[0, ..., :3]  # (B, H, W, 4) -> (H, W, 3)
    return rend


def render_sphere(image_size=256, num_samples=200, device=None):
    """
    Renders a sphere using parametric sampling. Samples num_samples ** 2 points.
    """

    if device is None:
        device = get_device()

    phi = torch.linspace(0, 2 * np.pi, num_samples)
    theta = torch.linspace(0, np.pi, num_samples)
    # Densely sample phi and theta on a grid
    Phi, Theta = torch.meshgrid(phi, theta)

    x = torch.sin(Theta) * torch.cos(Phi)
    y = torch.cos(Theta)
    z = torch.sin(Theta) * torch.sin(Phi)

    points = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=1)
    color = (points - points.min()) / (points.max() - points.min())

    sphere_point_cloud = pytorch3d.structures.Pointclouds(
        points=[points], features=[color],
    ).to(device)

    cameras = pytorch3d.renderer.FoVPerspectiveCameras(T=[[0, 0, 3]], device=device)
    renderer = get_points_renderer(image_size=image_size, device=device)
    rend = renderer(sphere_point_cloud, cameras=cameras)
    return rend[0, ..., :3].cpu().numpy()


def render_sphere_mesh(image_size=256, voxel_size=64, device=None):
    if device is None:
        device = get_device()
    min_value = -1.1
    max_value = 1.1
    X, Y, Z = torch.meshgrid([torch.linspace(min_value, max_value, voxel_size)] * 3)
    voxels = X ** 2 + Y ** 2 + Z ** 2 - 1
    vertices, faces = mcubes.marching_cubes(mcubes.smooth(voxels), isovalue=0)
    vertices = torch.tensor(vertices).float()
    faces = torch.tensor(faces.astype(int))
    # Vertex coordinates are indexed by array position, so we need to
    # renormalize the coordinate system.
    vertices = (vertices / voxel_size) * (max_value - min_value) + min_value
    textures = (vertices - vertices.min()) / (vertices.max() - vertices.min())
    textures = pytorch3d.renderer.TexturesVertex(vertices.unsqueeze(0))

    mesh = pytorch3d.structures.Meshes([vertices], [faces], textures=textures).to(
        device
    )
    lights = pytorch3d.renderer.PointLights(location=[[0, 0.0, -4.0]], device=device,)
    renderer = get_mesh_renderer(image_size=image_size, device=device)
    R, T = pytorch3d.renderer.look_at_view_transform(dist=3, elev=0, azim=180)
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
    rend = renderer(mesh, cameras=cameras, lights=lights)
    return rend[0, ..., :3].detach().cpu().numpy().clip(0, 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--render",
        type=str,
        default="rgbd",
        choices=["rgbd","point_cloud", "parametric", "implicit"],
    )
    parser.add_argument("--output_path", type=str, default="images/plant2.jpg")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--num_samples", type=int, default=100)
    args = parser.parse_args()
    if args.render == "rgbd":
        image = load_rgbd_data()
    elif args.render == "point_cloud":
        image = render_bridge(image_size=args.image_size)
    elif args.render == "parametric":
        image = render_sphere(image_size=args.image_size, num_samples=args.num_samples)
    elif args.render == "implicit":
        image = render_sphere_mesh(image_size=args.image_size)
    else:
        raise Exception("Did not understand {}".format(args.render))
    plt.imsave(args.output_path, image)