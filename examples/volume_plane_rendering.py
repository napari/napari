"""
Display one 3-D volume layer using the add_volume API and display it as a plane
with a simple widget for modifying plane parameters
"""
import napari
import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QWidget, QGroupBox, QVBoxLayout, QHBoxLayout
from skimage import data
from superqt import QLabeledDoubleSlider


viewer = napari.Viewer(ndisplay=3)

# add a volume
blobs = data.binary_blobs(
    length=64, volume_fraction=0.1, n_dim=3
).astype(float)
volume_layer = viewer.add_image(
    blobs, rendering='mip', name='volume', blending='additive', opacity=0.25
)

# add the same volume and render as plane
# plane should be in 'additive' blending mode or depth looks all wrong
plane_parameters = {
    'position': (32, 32, 32),
    'normal': (0, 1, 0),
    'thickness': 10,
}

plane_layer = viewer.add_image(
    blobs,
    rendering='average',
    name='plane',
    depiction='plane',
    blending='additive',
    opacity=0.5,
    plane=plane_parameters
)
viewer.axes.visible = True
viewer.camera.angles = (45, 45, 45)
viewer.camera.zoom = 5
napari.run()
