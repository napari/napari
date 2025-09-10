"""
Interactive scripting
=====================

.. tags:: interactivity
"""

import time

import numpy as np

import napari
from napari.qt import thread_worker
from napari.utils.task_status import task_status_manager

# create the viewer with an image
data = np.random.random((512, 512))
viewer = napari.Viewer()
layer = viewer.add_image(data)

def update_layer(data):
    layer.data = data

@thread_worker(connect={'yielded': update_layer})
def create_data(*, update_period, num_updates):
    # number of times to update
    for _k in range(num_updates):
        yield np.random.random((512, 512))
        time.sleep(update_period)

create_data(update_period=0.05, num_updates=50)

# prevent showing `ConfirmCloseDialog` even when there are task being done
# needed to prevent napari CI (tests and docs building) from getting stuck
def no_status():
    return []

task_status_manager.get_status = no_status

if __name__ == '__main__':
    napari.run()
