"""This example demonstrates how to execute a python script by drag'n'drop

To use this example, open napari and drag this file into the viewer.
"""
import sys

from napari import current_viewer
from napari.types import ImageData


def add_layers(img1: ImageData, img2: ImageData) -> ImageData:
    return img1 + img2

if current_viewer() is None:
    sys.exit(0)

current_viewer().open_sample('napari', 'cells3d')
current_viewer().window.add_function_widget(add_layers)

