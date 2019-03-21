# This is in interactive test

from time import sleep

import threading

from napari.components import Dims
from napari.components._dims.dims import DimsMode
from napari.components._dims.qtdims import QtDims
from napari.util import app_context

with app_context():

    # Instanciates a dimensions model:
    dims = Dims(3)

    dims.set_mode(0, DimsMode.Point)
    dims.set_point(1,50)
    dims.set_point(2,50)

    # defines a axis change listener:
    def listener_axis(event):
        axis = event.axis
        slicespec, projectspec = dims.slice_and_project
        print("axis: %d changed, slice: %s, project: %s" % (axis, slicespec[axis], projectspec[axis]))
        print("dims: %s" % event.source)

    # connects listener to model:
    dims.events.axis.connect(listener_axis)

    # defines a axis change listener:
    def listener_nbdim(event):
        print("dims changed from: "+str(event.source))

    # connects listener to model:
    dims.events.ndims.connect(listener_nbdim)

    # creates a widget to view (and control) the model:
    widget = QtDims(dims)

    # makes the view visible on the desktop:
    widget.show()


    # a loop to simulate changes to the model happening outside of the QT event loop:
    def myloop():
        for i in range(0, 1000):
            dims.set_point(0, i % 100)

            #print(widget.sliders[0])
            #print(widget.layout().itemAt(0))

            sleep(0.1)

            if i % 50 == 0:
                dims.num_dimensions = 3 + int(i // 50)

            if i>100 and i % 150 == 0:
                dims.num_dimensions = dims.num_dimensions - 1

            if not widget.isVisible():
                return


    # starts the thread for the loop:
    #threading.Thread(target=myloop).start()
