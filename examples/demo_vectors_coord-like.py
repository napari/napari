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
        N=1000
        pos = np.zeros((N, 4), dtype=np.float32)
        phi_space = np.linspace(0, 4*np.pi, N)
        radius_space = np.linspace(0, 100, N)

        #assign x-y position
        pos[:, 0] = radius_space*np.cos(phi_space)
        pos[:, 1] = radius_space*np.sin(phi_space)

        #assign x-y projection
        pos[:, 2] = radius_space*np.cos(phi_space)
        pos[:, 3] = radius_space*np.sin(phi_space)

        self.viewer.add_vectors(pos)

        self.win.show()


if __name__ == '__main__':
    # starting
    application = QApplication(sys.argv)

    viewer = Viewer()
    win = Window(Viewer(), show=False)

    multi_win = vector_window_example(win)

    sys.exit(application.exec_())