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

"""
Example script that creates a napari window and populates it with a random image and changing vectors

Uses pyqtSignal and Slots to communicate between processes

Uses QRunnable so that timing thread is separate from UI thread
"""
class MainWindow(QWidget):

    def __init__(self, window):
        super().__init__()
        self.win = window
        self.viewer = self.win.viewer

        image_random = np.random.rand(512,512)
        vect_random = np.random.rand(2, 4)
        self.layer1 = self.viewer.add_image(image_random, {})
        self.layer2 = self.viewer.add_vectors(vect_random)
        self.layer2.color = 'g'

        self.win.show()

    @pyqtSlot(np.ndarray)
    def update_image(self, data: np.ndarray):
        self.layer1.image = data

    @pyqtSlot(np.ndarray)
    def update_vectors(self, data: np.ndarray):
        self.layer2.vectors = data

    def make_connection(self, signal):
        if isinstance(signal, NewImageSender):
            print('connecting ImageSender to window slot')
            signal.new_image.connect(self.update_image)
        if isinstance(signal, NewVectorSender):
            print('connecting VectorSender to window slot')
            signal.new_vector.connect(self.update_vectors)

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

    def emit_image_sequence(self, num_images: int = 5, time_delay: float = 5):
        for k in range(0, num_images):
            self.data = np.random.rand(512,512)
            self.new_image.emit(self.data)
            time.sleep(time_delay)

    def start(self):
        self.p = ProcessRunnable(target=self.emit_image_sequence, args=(10,1))
        self.p.start()


class NewVectorSender(QObject, QRunnable):

    new_vector = pyqtSignal(np.ndarray)

    pos = np.random.rand(2, 4)

    def emit_vector_stream(self, time_delay: float = 0.01):
        while True:
            for k in range(0, 1000):
                time.sleep(time_delay)

                # sample vector coord-like data
                self.pos = np.zeros((k, 4), dtype=np.float32)
                phi_space = np.linspace(0, k*(4 * np.pi)/1000, k)
                radius_space = np.linspace(0, (k*100)/1000, k)

                # assign x-y position
                self.pos[:, 0] = radius_space * np.cos(phi_space) + 256
                self.pos[:, 1] = radius_space * np.sin(phi_space) + 256

                # assign x-y projection
                self.pos[:, 2] = radius_space * np.cos(phi_space)
                self.pos[:, 3] = radius_space * np.sin(phi_space)

                self.new_vector.emit(self.pos)

    def start(self):
        self.p = ProcessRunnable(target=self.emit_vector_stream, args=())
        self.p.start()

if __name__ == '__main__':
    # starting
    application = QApplication(sys.argv)

    viewer = Viewer()
    win = Window(Viewer(), show=True)

    custom_window = MainWindow(win)
    newim = NewImageSender()
    newvect = NewVectorSender()

    #connect signals
    custom_window.make_connection(newim)
    custom_window.make_connection(newvect)

    newim.start()
    newvect.start()

    sys.exit(application.exec_())