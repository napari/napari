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

    # sample vector image-like data
    # 50x25 grid of slanted lines
    n = 50
    m = 25
    pos = np.zeros(shape=(n, m, 2), dtype=np.float32)
    rand1 = np.random.random_sample(n * m)
    rand2 = np.random.random_sample(n * m)

    # assign projections for each vector
    pos[:, :, 0] = rand1.reshape((n, m))
    pos[:, :, 1] = rand2.reshape((n, m))

    # add the vectors
    viewer.add_vectors(pos)


