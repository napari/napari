"""
This example generates an image of vectors
Vector data is an array of shape (M, N, P, 3)
Each vector position is defined by an (x-proj, y-proj, z-proj) element
which are vector projections centered on a pixel of the MxNxP grid
"""

import napari
import numpy as np


with napari.gui_qt():
    # create the viewer and window
    viewer = napari.Viewer()

    m = 10
    n = 20
    p = 40

    image = 0.2 * np.random.random((m, n, p)) + 0.5
    layer = viewer.add_image(image, clim_range=[0, 1], name='background')

    # sample vector image-like data
    # n x m grid of slanted lines
    # random data on the open interval (-1, 1)
    pos = np.zeros(shape=(m, n, p, 3), dtype=np.float32)
    for i in range(3):
        pos[:, :, :, i] = (
            2 * (np.random.random_sample(n * m * p) - 0.5)
        ).reshape((m, n, p))

    print(image.shape, pos.shape)

    # add the vectors
    vect = viewer.add_vectors(pos, edge_width=0.2, length=2.5)
    viewer.dims.swap_display(0, 1)
