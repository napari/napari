#!/usr/bin/env python
# title           : this_python_file.py
# description     :This will create a header for a python script.
# author          :bryant.chhun
# date            :12/19/18
# version         :0.0
# usage           :python this_python_file.py -flags
# notes           :
# python_version  :3.6


import sys
from PyQt5.QtWidgets import QApplication, QWidget
from napari_gui import Window, Viewer

import numpy as np

from skimage import data


class open_multi(QWidget):

    def __init__(self, window):
        super().__init__()
        self.win = window
        self.viewer = self.win.viewer

        N = 200
        N2 = N * N
        pos = np.zeros((N2, 2), dtype=np.float32)
        dim = np.linspace(0, 4 * N - 1, N)
        xv, yv = np.meshgrid(dim, dim)
        pos[:, 0] = xv.flatten()
        pos[:, 1] = yv.flatten()

        # zigzag the lines
        pos[::2, 1] += (4 * N - 1) / N
        print(np.max(pos, axis=0))

        self.viewer.add_markers(pos)
        self.viewer.add_vectors(pos)
        self.viewer.add_markers(pos)

        self.win.show()


    def add_another_image(self):
        image_random = np.random.rand(512,512)
        self.viewer.add_image(image_random, {})
        self.win.show()

    def modify_image(self):
        self.win.show()
        # self.layers[-1].image = self.image3D_2

if __name__ == '__main__':
    # starting
    application = QApplication(sys.argv)

    viewer = Viewer()
    win = Window(Viewer(), show=False)

    multi_win = open_multi(win)
    # multi_win.add_another_image()
    # multi_win.add_another_image()
    # multi_win.add_another_image()

    # multi_win.modify_image()


    sys.exit(application.exec_())