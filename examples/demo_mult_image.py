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

        image_random = np.random.rand(512,512)

        # self.meta = dict(name='3D1C', itype='mono')
        # self.layers.append(self.viewer.add_image(self.image3D, self.meta))
        # self.layers[0].cmap = 'blues'
        # self.layers[0].interpolation = 'spline36'

        self.viewer.add_image(image_random, {})
        self.viewer.add_image(data.astronaut(), {})

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
    multi_win.add_another_image()
    # multi_win.add_another_image()
    # multi_win.add_another_image()
    # multi_win.add_another_image()

    # multi_win.modify_image()


    sys.exit(application.exec_())