"""
This example generates an image of vectors
Vector data is an array of shape (N, M, 2)
Each vector position is defined by an (x-proj, y-proj) element
    where x-proj and y-proj are the vector projections at each center
    where each vector is centered on a pixel of the NxM grid
"""

import sys
from PyQt5.QtWidgets import QApplication
from napari_gui import Window, Viewer

import numpy as np

# starting
application = QApplication(sys.argv)

# create the viewer and window
viewer = Viewer()
win = Window(viewer)

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

sys.exit(application.exec_())

