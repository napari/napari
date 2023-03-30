"""
Surface with texture and vertex_colors
======================================

Display a 3D surface with both texture and color maps.

.. tags:: visualization-nD
"""

import numpy as np
from vispy.io import imread, load_data_file, read_mesh

import napari

# create the viewer and window
viewer = napari.Viewer(ndisplay=3)

# load the model and texture
mesh_path = load_data_file('spot/spot.obj.gz')
vertices, faces, _normals, texcoords = read_mesh(mesh_path)
n = len(vertices)
texture_path = load_data_file('spot/spot.png')
texture = np.flipud(imread(texture_path))

np.random.seed(0)
viewer.add_surface(
    (vertices, faces, np.random.random((3, 3, n))),
    texture=texture,
    texcoords=texcoords,
    colormap="plasma",
    shading="smooth",
    name="vertex_values and texture",
)
viewer.add_surface(
    (vertices, faces),
    translate=(1, 0, 0),
    texture=texture,
    texcoords=texcoords,
    shading="flat",
    name="texture only",
)
viewer.add_surface(
    (vertices, faces),
    translate=(-1, 0, 0),
    texture=texture,
    texcoords=texcoords,
    # the vertices are _roughly_ in [-1, 1] for this model and RGB values just
    # get clipped to [0, 1], adding 0.5 brightens it up a little :)
    vertex_colors=vertices + 0.5,
    shading="none",
    name="vertex_colors and texture",
)

viewer.camera.angles = (25.0, -50.0, -125.0)
viewer.camera.zoom = 150


if __name__ == '__main__':
    napari.run()
