"""
Display one 3-D volume layer using the add_volume API and display it as a plane
with a simple widget for modifying plane parameters
"""
import napari
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QWidget, QGroupBox, QVBoxLayout, QHBoxLayout, QPushButton
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
    plane=plane_parameters
)


class PlaneWidget(QWidget):
    def __init__(self):
        super().__init__()

        master_layout = QVBoxLayout(self)

        self.plane_orientation_groupbox = QGroupBox('plane orientation')
        self.z = QPushButton(text='z')
        self.y = QPushButton(text='y')
        self.x = QPushButton(text='x')
        self.oblique = QPushButton(text='oblique')

        plane_orientation_layout = QHBoxLayout(self.plane_orientation_groupbox)
        for button in (self.z, self.y, self.x, self.oblique):
            plane_orientation_layout.addWidget(button)

        self.thickness_box = QGroupBox('plane thickness')
        self.thickness_spinbox = QLabeledDoubleSlider(Qt.Horizontal, self)
        self.thickness_spinbox.setMinimum(1.0)
        self.thickness_spinbox.setMaximum(64)
        self.thickness_spinbox.setValue(10)

        thickness_layout = QHBoxLayout(self.thickness_box)
        thickness_layout.addWidget(self.thickness_spinbox)

        master_layout.addWidget(self.plane_orientation_groupbox)
        master_layout.addWidget(self.thickness_box)


def update_plane_thickness(widget):
    plane_layer.embedded_plane.thickness = widget.thickness_spinbox.value()


def set_plane_normal(normal_vector):
    plane_layer.embedded_plane.normal = normal_vector


def create_plane_widget():
    widget = PlaneWidget()
    widget.x.clicked.connect(lambda: set_plane_normal((0, 0, 1)))
    widget.y.clicked.connect(lambda: set_plane_normal((0, 1, 0)))
    widget.z.clicked.connect(lambda: set_plane_normal((1, 0, 0)))
    widget.oblique.clicked.connect(lambda: set_plane_normal((1, 1, 1)))
    widget.thickness_spinbox.valueChanged.connect(
        lambda: update_plane_thickness(widget)
    )
    return widget


plane_widget = create_plane_widget()
viewer.window.add_dock_widget(
    plane_widget, name='Plane Widget', area='left'
)

viewer.axes.visible = True
viewer.camera.angles = (100, 20, 120)
viewer.camera.zoom = 5
viewer.text_overlay.visible = True
viewer.text_overlay.text = "click and drag the plane to shift it along its normal vector"

napari.run()
