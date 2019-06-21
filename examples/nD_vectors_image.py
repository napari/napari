"""
This example generates an image of vectors
Vector data is an array of shape (M, N, P, 3)
Each vector position is defined by an (x-proj, y-proj, z-proj) element
which are vector projections centered on a pixel of the MxNxP grid
"""

import napari
from napari.util import app_context

import numpy as np

with app_context():
    # create the viewer and window
    viewer = napari.Viewer()

    n = 40
    m = 60
    p = 80

    image = 0.2 * np.random.random((n, m, p)) + 0.5
    layer = viewer.add_image(image, clim_range=[0, 1], name='background')
    layer.colormap = 'gray'

    # sample vector image-like data
    # n x m grid of slanted lines
    # random data on the open interval (-1, 1)
    pos = np.zeros(shape=(n, m, p, 3), dtype=np.float32)
    rand1 = 2 * (np.random.random_sample(n * m * p) - 0.5)
    rand2 = 2 * (np.random.random_sample(n * m * p) - 0.5)

    # assign projections for each vector
    pos[:, :, :, 0] = 0
    pos[:, :, :, 1] = rand1.reshape((n, m, p))
    pos[:, :, :, 2] = rand2.reshape((n, m, p))

    print(image.shape, pos.shape)

    # add the vectors
    vect = viewer.add_vectors(pos, width=0.2, length=2.5)
