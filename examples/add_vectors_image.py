"""
This example generates an image of vectors
Vector data is an array of shape (N, M, 2)
Each vector position is defined by an (x-proj, y-proj) element
    where x-proj and y-proj are the vector projections at each center
    where each vector is centered on a pixel of the NxM grid
"""

from napari import ViewerApp
from napari.util import app_context

import numpy as np

with app_context():
    # create the viewer and window
    viewer = ViewerApp()

    n = 100
    m = 200

    image = 0.2*np.random.random((n, m)) + 0.5
    layer = viewer.add_image(image, clim_range=[0, 1], name='background')
    layer.colormap = 'gray'

    # sample vector image-like data
    # n x m grid of slanted lines
    # random data on the open interval (-1, 1)
    pos = np.zeros(shape=(m, n, 2), dtype=np.float32)
    rand1 = 2*(np.random.random_sample(n * m)-0.5)
    rand2 = 2*(np.random.random_sample(n * m)-0.5)

    # assign projections for each vector
    pos[:, :, 0] = rand1.reshape((m, n))
    pos[:, :, 1] = rand2.reshape((m, n))

    # add the vectors
    vect = viewer.add_vectors(pos, width=0.2, length=2.5)
