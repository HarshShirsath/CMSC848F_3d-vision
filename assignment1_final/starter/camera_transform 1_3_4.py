import argparse
import os
import matplotlib.pyplot as plt
import pytorch3d
import torch
import numpy as np

from utils import get_device, get_mesh_renderer


def render_cow_transformed(
    cow_path="data/cow_with_axis.obj",
    image_size=512,
    R_relative=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    T_relative=[0, 0, 0],
    output_path="images/transformed_cow.jpg",
    device=None,
):
    if device is None:
        device = get_device()
    meshes = pytorch3d.io.load_objs_as_meshes([cow_path]).to(device)

    # Create batched rotation and translation matrices
    num_cams = 1
    R_relative = torch.tensor(R_relative, dtype=torch.float32).unsqueeze(0).expand(num_cams, -1, -1)
    T_relative = torch.tensor(T_relative, dtype=torch.float32).unsqueeze(0).expand(num_cams, -1)

    # No need to change this part
    renderer = get_mesh_renderer(image_size=image_size, device=device)
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(
        R=R_relative, T=T_relative, device=device,
    )
    lights = pytorch3d.renderer.PointLights(location=[[0, 0.0, -3.0]], device=device,)
    rend = renderer(meshes, cameras=cameras, lights=lights)
    rend_image = rend[0, ..., :3].cpu().numpy()

    # Save the rendered image
    plt.imsave(output_path, rend_image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cow_path", type=str, default="data/cow_with_axis.obj")
    parser.add_argument("--image_size", type=int, default=512)
    args = parser.parse_args()

    # Render different transformations
    render_cow_transformed(
        cow_path=args.cow_path,
        image_size=args.image_size,
        R_relative=[[0, -1, 0], [-1, 0, 0], [0, 0, 1]],  # 90-degree clockwise rotation
        T_relative=[0, 0, 3],  # No translation_done
        output_path="images/transform 1.jpg",
    )

    # render_cow_transformed(
    #     cow_path=args.cow_path,
    #     image_size=args.image_size,
    #     R_relative=[[0, 0, 1], [0, 1, 0], [-1, 0, 0]],  # Slight rotation in Y
    #     T_relative=[0, 0, 6],  # No translation
    #     output_path="images/transform 2.jpg",
    # )

    render_cow_transformed(
        cow_path=args.cow_path,
        image_size=args.image_size,
        R_relative=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],  # No rotation
        T_relative=[0, 0, 4.0],  # Zoom-in_Done
        output_path="images/transform 3.jpg",
    )

    render_cow_transformed(
        cow_path=args.cow_path,
        image_size=args.image_size,
        R_relative=[[0, 0, -2], [0, 1, 0], [-1, 0, 0]],  # Side view
        T_relative=[0, 0, 4],  # No translation_Done
        output_path="images/transform 4.jpg",
    )
