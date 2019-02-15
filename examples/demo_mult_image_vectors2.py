#!/usr/bin/env python
# title           : demo_mult_image_vectors.py
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

class vector_window_example(QWidget):
    '''
    This example generates an image of vectors
    The end points of the vectors are marked by markers
    '''

    def __init__(self, window):
        super().__init__()
        self.win = window
        self.viewer = self.win.viewer

        # sample vector data
        N = 6
        pos = np.zeros((N*N, 2), dtype=np.float32)
        xdim = np.linspace(0, N, N)
        ydim = np.linspace(0, N, N)
        xv, yv = np.meshgrid(xdim, ydim)
        pos[:, 0] = xv.flatten()
        pos[:, 1] = yv.flatten()

        # add slant to the lines
        # add a value to every other y-coordinate
        pos[1::2, 1] += 0.5

        # mk1 = self.viewer.add_markers(pos)
        # mk1.size = 0.3
        vect1 = self.viewer.add_vectors(pos)
        vect1.width = 4
        vect1.arrow_size = 100
        vect1.arrows = pos.reshape((len(pos)//2, 4)) + 0.5
        # vect1.arrows = None

        self.win.show()


if __name__ == '__main__':
    # starting
    application = QApplication(sys.argv)

    viewer = Viewer()
    win = Window(Viewer(), show=False)

    multi_win = vector_window_example(win)

    sys.exit(application.exec_())