"""
Display a 2D surface
"""

import numpy as np
import napari


data = np.array([[0, 0], [0, 20], [10, 0], [10, 10]])
faces = np.array([[0, 1, 2], [1, 2, 3]])
values = np.linspace(0, 1, len(data))

# add the surface
viewer = napari.view_surface((data, faces, values))

napari.run()
