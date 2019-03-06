# This is in interactive test

import sys
from time import sleep

import threading
from PyQt5.QtWidgets import QApplication

from napari.components import Dims
from napari.components._dims.view import QtDims

# starts the QT event loop
app = QApplication(sys.argv)

# Instanciates a dimensions model:
dims = Dims(3)


# defines a model listener:
def listener(source, axis):
    print((source, axis))


# adds listener to model:
dims.add_listener(listener)

# creates a widget to view (and control) the model:
widget = QtDims(dims)

# makes the view visible on the desktop:
widget.show()


# a loop to simulate changes to the model happening outside of the QT event loop:
def myloop():
    for i in range(0, 1000):
        dims.set_point(0, i % 100)
        sleep(0.1)


# starts the thread for the loop:
threading.Thread(target=myloop).start()

# Start the QT event loop.
sys.exit(app.exec())
