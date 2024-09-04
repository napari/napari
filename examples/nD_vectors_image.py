"""
nD vectors image
================

This example generates an image of vectors
Vector data is an array of shape (M, N, P, 3)
Each vector position is defined by an (x-proj, y-proj, z-proj) element
which are vector projections centered on a pixel of the MxNxP grid

.. tags:: visualization-nD
"""

import numpy as np

import napari

# create the viewer and window
viewer = napari.Viewer()

m = 10
n = 20
p = 40

image = 0.2 * np.random.random((m, n, p)) + 0.5
layer = viewer.add_image(image, contrast_limits=[0, 1], name='background')

# sample vector image-like data
# n x m grid of slanted lines
# random data on the open interval (-1, 1)
pos = np.random.uniform(-1, 1, size=(m, n, p, 3))
print(image.shape, pos.shape)

# add the vectors
vect = viewer.add_vectors(pos, edge_width=0.2, length=2.5)

if __name__ == '__main__':
    napari.run()
