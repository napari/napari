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
    'enabled': True
}

plane_layer = viewer.add_image(
    blobs,
    rendering='average',
    name='plane',
    blending='additive',
    opacity=0.5,
    experimental_slicing_plane=plane_parameters
)

class PlaneWidget(QWidget):
    def __init__(self):
        super().__init__()

        master_layout = QVBoxLayout(self)

        self.position_slider_box = QGroupBox('plane position (axis 1)')
        self.position_slider = QLabeledDoubleSlider(Qt.Horizontal, self)
        self.position_slider.setMinimum(0.05)
        self.position_slider.setMaximum(64)
        self.position_slider.setValue(32)

        position_layout = QHBoxLayout(self.position_slider_box)
        position_layout.addWidget(self.position_slider)

        self.thickness_box = QGroupBox('plane thickness')
        self.thickness_spinbox = QLabeledDoubleSlider(Qt.Horizontal, self)
        self.thickness_spinbox.setMinimum(1.0)
        self.thickness_spinbox.setMaximum(64)
        self.thickness_spinbox.setValue(10)

        thickness_layout = QHBoxLayout(self.thickness_box)
        thickness_layout.addWidget(self.thickness_spinbox)

        master_layout.addWidget(self.position_slider_box)
        master_layout.addWidget(self.thickness_box)


def update_plane_y_position(widget):
    plane_position = [32, widget.position_slider.value(), 32]
    viewer.layers['plane'].experimental_slicing_plane.position = plane_position


def update_plane_thickness(widget):
    plane_layer.experimental_slicing_plane.thickness = widget.thickness_spinbox.value()


def create_plane_widget():
    widget = PlaneWidget()
    widget.position_slider.valueChanged.connect(
        lambda: update_plane_y_position(widget)
    )
    widget.thickness_spinbox.valueChanged.connect(
        lambda: update_plane_thickness(widget)
    )
    return widget


plane_widget = create_plane_widget()
viewer.window.add_dock_widget(
    plane_widget, name='Plane Widget', area='left'
)
viewer.axes.visible = True
viewer.camera.angles = (45, 45, 45)
viewer.camera.zoom = 5
napari.run()
