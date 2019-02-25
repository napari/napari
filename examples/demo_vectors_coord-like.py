#!/usr/bin/env python
# title           : demo_vectors_coord-like.py
# description     :demonstration of vector image using napari layers
# author          :bryant.chhun
# date            :1/16/19
# version         :0.0
# usage           :
# notes           :
# python_version  :3.6


import sys
from PyQt5.QtWidgets import QApplication, QWidget
from napari_gui import Window, Viewer

import numpy as np

'''
This example generates an image of vectors
Vector data is an array of shape (N, 4)
Each vector position is defined by an (x, y, x-proj, y-proj) element
    where x and y are the center points
    where x-proj and y-proj are the vector projections at each center

'''
class vector_window_example(QWidget):


    def __init__(self, window):
        super().__init__()
        self.win = window
        self.viewer = self.win.viewer

        # sample vector coord-like data
        # 10x10 grid of slanted lines
        N = 11
        pos = np.zeros((N*N, 4), dtype=np.float32)
        xdim = np.linspace(0, N, N)
        ydim = np.linspace(0, N, N)
        xv, yv = np.meshgrid(xdim, ydim)
        pos[:, 0] = xv.flatten()
        pos[:, 1] = yv.flatten()

        # all vectors have same projection
        pos[:, 2] = 0.5
        pos[:, 3] = 0.5
        # radial vectors from top center
        # pos[:, 2] = pos[:, 0] - pos[5:6, 0]
        # pos[:, 3] = pos[:, 1] - pos[5:6, 1]

        self.viewer.add_vectors(pos)

        self.win.show()


if __name__ == '__main__':
    # starting
    application = QApplication(sys.argv)

    viewer = Viewer()
    win = Window(Viewer(), show=False)

    multi_win = vector_window_example(win)

    sys.exit(application.exec_())