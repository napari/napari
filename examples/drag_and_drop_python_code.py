"""
Drag and Drop Python Code Example
=================================

This example demonstrates how to execute a python script by drag'n'drop

To use this example, open napari and drag this file into the viewer.

.. tags:: interactivity
"""
from napari import Viewer
from napari.types import ImageData


def add_layers(img1: ImageData, img2: ImageData) -> ImageData:
    return img1 + img2

viewer = Viewer()

viewer.open_sample('napari', 'cells3d')
viewer.window.add_function_widget(add_layers)

