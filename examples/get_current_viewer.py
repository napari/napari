"""
Get current viewer
==================

Get a reference to the current napari viewer.

Whilst this example is contrived, it can be useful to get a reference to the
viewer when the viewer is out of scope.

.. tags:: gui
"""
import numpy as np

import napari

# create viewer
viewer = napari.Viewer()

# lose reference to viewer
viewer = 'oops no viewer here'

# get that reference again
viewer = napari.current_viewer()

# work with the viewer
x = np.arange(256)
y = np.arange(256).reshape((256, 1))
# from: https://botsin.space/@bitartbot/113553754823363986
image = (-(~((y - x) ^ (y + x)))) % 11
layer = viewer.add_image(image)
layer.contrast_limits = (8.5, 10)

if __name__ == '__main__':
    napari.run()
