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
Vector data is an array of shape (N, M, 2)
Each vector position is defined by an (x-proj, y-proj) element
    where x-proj and y-proj are the vector projections at each center
    where each vector is centered on a pixel of the NxM grid
'''

class vector_window_example(QWidget):


    def __init__(self, window):
        super().__init__()
        self.win = window
        self.viewer = self.win.viewer

        # sample vector image-like data
        # 50x25 grid of slanted lines
        N = 50
        M = 25
        pos = np.zeros(shape=(N, M, 2), dtype=np.float32)
        rand1 = np.random.random_sample(N*M)
        rand2 = np.random.random_sample(N*M)
        # assign projections for each vector
        pos[:, :, 0] = rand1.reshape((N,M))
        pos[:, :, 1] = rand2.reshape((N,M))

        self.viewer.add_vectors(pos)

        self.win.show()


if __name__ == '__main__':
    # starting
    application = QApplication(sys.argv)

    viewer = Viewer()
    win = Window(Viewer(), show=False)

    multi_win = vector_window_example(win)

    sys.exit(application.exec_())