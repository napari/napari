"""
Progress bar minimal
====================

This file provides minimal working examples of progress bars in
the napari viewer.

.. tags:: gui
"""

from random import choice
from time import sleep

import numpy as np
from qtpy.QtWidgets import QPushButton, QVBoxLayout, QWidget

import napari
from napari.utils import progress


def process(im_slice):
    # do something with your image slice
    sleep(0.4)

def iterable():
    """using progress as a wrapper for iterables
    """
    my_stacked_volume = np.random.random((5, 4, 500, 500))
    # we can wrap any iterable object in `progress` and see a progress
    # bar in the viewer
    for im_slice in progress(my_stacked_volume):
        process(im_slice)

def iterable_w_context():
    """using progress with a context manager
    """
    my_stacked_volume = np.random.random((5, 4, 500, 500))
    # progress provides a context manager we can use for automatic
    # teardown of our widget once iteration is complete. Wherever
    # possible, we should *always* use progress within a context
    with progress(my_stacked_volume) as pbr:
        for i, im_slice in enumerate(pbr):
            # using a context manager also allows us to manipulate
            # the progress object e.g. by setting a description
            pbr.set_description(f"Slice {i}")

            # we can group progress bars together in the viewer
            # by passing a parent progress bar to new progress
            # objects' nest_under attribute
            for channel in progress(im_slice, nest_under=pbr):
                process(channel)

def indeterminate():
    """By passing a total of 0, we can have an indeterminate progress bar
    """

    # note progress(total=0) is equivalent to progress()
    with progress(total=0) as pbr:
        x = 0
        while x != 42:
            pbr.set_description(f"Processing {x}")
            x = choice(range(100))
            sleep(0.05)

def arbitrary_steps():
    """We can manually control updating the value of the progress bar.
    """
    with progress(total=4) as pbr:
        sleep(3)
        pbr.set_description("Step 1 Complete")
        # manually updating the progress bar by 1
        pbr.update(1)

        sleep(1)
        pbr.set_description("Step 2 Complete")
        pbr.update(1)

        sleep(2)
        pbr.set_description("Processing Complete!")
        # we can manually update by any number of steps
        pbr.update(2)

        # sleeping so we can see full completion
        sleep(1)

viewer = napari.Viewer()
button_layout = QVBoxLayout()

iterable_btn = QPushButton("Iterable")
iterable_btn.clicked.connect(iterable)
button_layout.addWidget(iterable_btn)

iterable_context_btn = QPushButton("Iterable With Context")
iterable_context_btn.clicked.connect(iterable_w_context)
button_layout.addWidget(iterable_context_btn)

indeterminate_btn = QPushButton("Indeterminate")
indeterminate_btn.clicked.connect(indeterminate)
button_layout.addWidget(indeterminate_btn)

steps_btn = QPushButton("Arbitrary Steps")
steps_btn.clicked.connect(arbitrary_steps)
button_layout.addWidget(steps_btn)

pbar_widget = QWidget()
pbar_widget.setLayout(button_layout)
pbar_widget.setObjectName("Progress Examples")

viewer.window.add_dock_widget(pbar_widget)
# showing the activity dock so we can see the progress bars
viewer.window._status_bar._toggle_activity_dock(True)

if __name__ == '__main__':
    napari.run()
