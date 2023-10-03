# import argparse
# import matplotlib.pyplot as plt
# import numpy as np
# import imageio

# def generate_torus_parametric(
#     major_radius=1.0, minor_radius=0.3, num_samples=200,
# ):
#     phi = np.linspace(0, 2 * np.pi, num_samples)
#     theta = np.linspace(0, 2 * np.pi, num_samples)
#     Phi, Theta = np.meshgrid(phi, theta)

#     # Adjust the equations for a spherical torus
#     x = (major_radius + minor_radius * np.cos(Theta)) * np.cos(Phi)
#     y = (major_radius + minor_radius * np.cos(Theta)) * np.sin(Phi)
#     z = minor_radius * np.sin(Theta)
    
#     return x, y, z

# def render_torus_parametric_gif(num_samples=200, duration=5):
#     x, y, z = generate_torus_parametric(num_samples=num_samples)

#     images = []

#     for azimuth in range(0, 360, 5):
#         fig = plt.figure()
#         ax = fig.add_subplot(111, projection='3d')
#         ax.plot_surface(x, y, z, cmap='viridis', edgecolor='none')
#         ax.view_init(elev=30, azim=azimuth)
#         ax.axis('off')
#         plt.tight_layout()

#         # Save the current frame
#         filename = f"frame_{azimuth:03d}.png"
#         plt.savefig(filename, dpi=100)
#         plt.close()

#         images.append(imageio.imread(filename))

#     # Save the images as a gif
#     imageio.mimsave("images/torus_parametric.gif", images, duration=duration)

# if __name__ == "__main__":
#     render_torus_parametric_gif()

import argparse
import matplotlib.pyplot as plt
import numpy as np
import imageio

def generate_torus_parametric(
    major_radius=1.0, minor_radius=0.3, num_samples=200,
):
    phi = np.linspace(0, 2 * np.pi, num_samples)
    theta = np.linspace(0, 2 * np.pi, num_samples)
    Phi, Theta = np.meshgrid(phi, theta)

    # Adjust the equations for a spherical torus
    x = (major_radius + minor_radius * np.cos(Theta)) * np.cos(Phi)
    y = (major_radius + minor_radius * np.cos(Theta)) * np.sin(Phi)
    z = minor_radius * np.sin(Theta)
    
    return x, y, z

def render_torus_parametric_gif(num_samples=200, duration=5):
    x, y, z = generate_torus_parametric(num_samples=num_samples)

    images = []

    for azimuth in range(0, 360, 5):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(x, y, z, cmap='viridis', edgecolor='none')
        ax.view_init(elev=30, azim=azimuth)
        ax.axis('off')
        plt.tight_layout()

        # Save the current frame
        filename = f"frame_{azimuth:03d}.png"
        plt.savefig(filename, dpi=100)
        plt.close()

        images.append(imageio.imread(filename))

    # Save the images as a gif
    imageio.mimsave("images/torus_parametric.gif", images, duration=duration)

if __name__ == "__main__":
    render_torus_parametric_gif()
