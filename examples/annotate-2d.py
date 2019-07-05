"""
Display one points layer ontop of one image layer using the add_points and
add_image APIs
"""

import numpy as np
from skimage import data
import napari
from napari.util import gui_qt

print("click to add points; close the window when finished.")

with gui_qt():
    viewer = napari.view(data.astronaut(), multichannel=True)
    points = viewer.add_points(np.zeros((0, 2)))
    points.mode = 'add'

print("you clicked on:")
print(points.coords)
