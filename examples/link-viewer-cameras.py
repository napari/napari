"""
Link cameras between two viewers
================================

This lets you zoom in on different layers side by side instead of overlaying
them. [1]_

.. [1] https://forum.image.sc/t/zooming-in-on-the-same-region-on-two-different-layers/95659/5

.. tags:: gui, layers, visualization-advanced, multicanvas
"""
from skimage import data, filters

import napari

# example data
coins = data.coins()
binary = coins > filters.threshold_otsu(coins)

# make two viewers
viewer0 = napari.Viewer()
viewer1 = napari.Viewer()

# add a layer to each
viewer0.add_image(coins)
viewer1.add_labels(binary)

# note: check available events with:
# viewer0.camera.events.emitters

# reusable functions to set camera zoom and center
def set_zoom(camera, value):
    camera.zoom = value

def set_center(camera, value):
    camera.center = value

# hook everything up
viewer0.camera.events.zoom.connect(
    lambda ev: set_zoom(viewer1.camera, ev.value)
)
viewer1.camera.events.zoom.connect(
    lambda ev: set_zoom(viewer0.camera, ev.value)
)
viewer0.camera.events.center.connect(
    lambda ev: set_center(viewer1.camera, ev.value)
)
viewer1.camera.events.center.connect(
    lambda ev: set_center(viewer0.camera, ev.value)
)

if __name__ == '__main__':
    napari.run()
