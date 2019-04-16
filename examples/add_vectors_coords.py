"""
This example generates an image of vectors
Vector data is an array of shape (N, 4)
Each vector position is defined by an (x, y, x-proj, y-proj) element
    where x and y are the center points
    where x-proj and y-proj are the vector projections at each center

"""

from napari import ViewerApp
from napari.util import app_context
from skimage import data

import numpy as np


with app_context():
    # create the viewer and window
    viewer = ViewerApp()

    layer = viewer.add_image(data.camera(), name='photographer')
    layer.colormap = 'gray'

    # sample vector coord-like data
    n = 1000
    pos = np.zeros((n, 4), dtype=np.float32)
    phi_space = np.linspace(0, 4*np.pi, n)
    radius_space = np.linspace(0, 100, n)

    # assign x-y position
    pos[:, 0] = radius_space*np.cos(phi_space) + 256
    pos[:, 1] = radius_space*np.sin(phi_space) + 256

    # assign x-y projection
    pos[:, 2] = 2*radius_space*np.cos(phi_space)
    pos[:, 3] = 2*radius_space*np.sin(phi_space)

    # add the vectors
    layer = viewer.add_vectors(pos, width=0.4)
