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


def myavgmethod(event):
    print("custom calculation of average")


def mylenmethod(event):
    print('custom calculation of length')


with app_context():
    # create the viewer and window
    viewer = ViewerApp()

    # sample vector image-like data
    # n x m grid of slanted lines
    # random data on the open interval (-1, 1)
    n = 100
    m = 50
    pos = np.zeros(shape=(n, m, 2), dtype=np.float32)
    rand1 = 2*(np.random.random_sample(n * m)-0.5)
    rand2 = 2*(np.random.random_sample(n * m)-0.5)

    # assign projections for each vector
    pos[:, :, 0] = rand1.reshape((n, m))
    pos[:, :, 1] = rand2.reshape((n, m))

    # add the vectors
    vect = viewer.add_vectors(pos)
    # vect.averaging_bind_to(myavgmethod)
    # vect.length_bind_to(mylenmethod)


