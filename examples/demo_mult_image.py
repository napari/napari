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

        image_random = np.random.rand(512, 512)

        self.viewer.add_image(image_random, {})
        self.viewer.add_image(data.astronaut(), {})

        self.win.show()

    def add_another_image(self):
        image_random = np.random.rand(512, 512)
        self.viewer.add_image(image_random, {})
        self.win.show()


if __name__ == '__main__':
    # starting
    application = QApplication(sys.argv)

    viewer = Viewer()
    win = Window(Viewer(), show=False)

    multi_win = open_multi(win)
    multi_win.add_another_image()

    sys.exit(application.exec_())
