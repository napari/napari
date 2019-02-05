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
from PyQt5.QtCore import pyqtSignal, pyqtSlot, QObject, QRunnable, QThreadPool
from PyQt5.QtWidgets import QApplication, QWidget
from napari_gui import Window, Viewer

import numpy as np
import time


class MainWindow(QWidget):

    def __init__(self, window):
        super().__init__()
        self.win = window
        self.viewer = self.win.viewer

        image_random = np.random.rand(512,512)
        self.layer1 = self.viewer.add_image(image_random, {})

        self.win.show()

    @pyqtSlot(np.ndarray)
    def update_image(self, data: np.ndarray):
        self.layer1.image = data

    def make_connection(self, signal):
        if isinstance(signal, NewImageSender):
            print('connecting sender to window slot')
            signal.new_image.connect(self.update_image)


class ProcessRunnable(QRunnable):
    def __init__(self, target, args):
        QRunnable.__init__(self)
        self.t = target
        self.args = args

    def run(self):
        self.t(*self.args)

    def start(self):
        QThreadPool.globalInstance().start(self)

class NewImageSender(QObject, QRunnable):

    new_image = pyqtSignal(np.ndarray)

    data = np.random.rand(512,512)

    def emit_image_sequence(self, num_images: int = 5, time_delay:float = 5):
        for k in range(0, num_images):
            self.data = np.random.rand(512,512)
            self.new_image.emit(self.data)
            time.sleep(time_delay)

    def start(self):
        self.p = ProcessRunnable(target=self.emit_image_sequence, args=(10,1))
        self.p.start()


if __name__ == '__main__':
    # starting
    application = QApplication(sys.argv)

    viewer = Viewer()
    win = Window(Viewer(), show=True)

    custom_window = MainWindow(win)
    sender = NewImageSender()

    #connect signals
    custom_window.make_connection(sender)

    # sender.run(5, 3)
    sender.start()

    sys.exit(application.exec_())