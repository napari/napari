"""
Display one 3-D volume layer using the add_volume API and display it as a plane
with a widget for modifying plane parameters
"""
import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QWidget, QGroupBox, QVBoxLayout, QHBoxLayout, QPushButton
from scipy.spatial.transform import Rotation as R
from skimage import data
from superqt import QLabeledDoubleSlider

import napari

blobs = data.binary_blobs(length=64, volume_fraction=0.1, n_dim=3).astype(
    float
)
viewer = napari.Viewer(ndisplay=3)
viewer.axes.visible = True

# add a volume
volume_layer = viewer.add_image(
    blobs, rendering='mip', name='volume', blending='additive', opacity=0.5
)

# add the same volume and render as plane
# plane should be in 'additive' blending mode or depth looks all wrong
plane_layer = viewer.add_image(
    blobs, rendering='average', name='plane', blending='additive', opacity=0.5,
)
plane_layer.render_as_plane = True

plane_layer.plane.position = (32, 32, 32)
plane_layer.plane.normal_vector = (1, 0, 0)
plane_layer.plane.thickness = 5

# add a point to display plane point
plane_point_layer = viewer.add_points(np.array([32, 32, 32]), face_color='cornflowerblue',
                                      name='plane point')

# add a vector to display the plane normal
normal_data = np.array(
    [[32, 32, 32],
     [1, 0, 0]]
).reshape((-1, 2, 3))
plane_normal_layer = viewer.add_vectors(normal_data, edge_color='cornflowerblue', length=15,
                                        edge_width=3, name='plane normal')

# add a point at each corner of the volume to help orient the user
bounding_box = np.array(
    [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 1],
    ]
) * 64
bounding_box_layer = viewer.add_points(bounding_box, face_color='seagreen', name='bounding_box')


class PlaneWidget(QWidget):
    def __init__(self):
        super().__init__()

        master_layout = QVBoxLayout(self)

        self.position_sliders = QGroupBox('plane position sliders')
        position_layout = QHBoxLayout(self.position_sliders)

        self.position_0 = QLabeledDoubleSlider(self)
        self.position_1 = QLabeledDoubleSlider(self)
        self.position_2 = QLabeledDoubleSlider(self)

        position_layout.addWidget(self.position_0)
        position_layout.addWidget(self.position_1)
        position_layout.addWidget(self.position_2)

        for slider in self.position_0, self.position_1, self.position_2:
            slider.setMinimum(0)
            slider.setMaximum(64)
            slider.setValue(32)

        self.orientation_presets = QGroupBox('plane orientation presets')
        orientation_presets_layout = QHBoxLayout(self.orientation_presets)

        self.normal_0_button = QPushButton('0', self)
        self.normal_1_button = QPushButton('1', self)
        self.normal_2_button = QPushButton('2', self)
        self.normal_oblique_button = QPushButton('oblique', self)

        orientation_presets_layout.addWidget(self.normal_0_button)
        orientation_presets_layout.addWidget(self.normal_1_button)
        orientation_presets_layout.addWidget(self.normal_2_button)
        orientation_presets_layout.addWidget(self.normal_oblique_button)

        self.orientation_controls = QGroupBox('plane orientation controls')
        orientation_controls_layout = QVBoxLayout(self.orientation_controls)

        self.normal_rotate_0_positive_button = QPushButton('R(0)+', self)
        self.normal_rotate_0_negative_button = QPushButton('R(0)-', self)
        self.normal_rotate_1_positive_button = QPushButton('R(1)+', self)
        self.normal_rotate_1_negative_button = QPushButton('R(1)-', self)
        self.normal_rotate_2_positive_button = QPushButton('R(2)+', self)
        self.normal_rotate_2_negative_button = QPushButton('R(2)-', self)

        orientation_controls_layout.addWidget(self.normal_rotate_0_positive_button)
        orientation_controls_layout.addWidget(self.normal_rotate_0_negative_button)
        orientation_controls_layout.addWidget(self.normal_rotate_1_positive_button)
        orientation_controls_layout.addWidget(self.normal_rotate_1_negative_button)
        orientation_controls_layout.addWidget(self.normal_rotate_2_positive_button)
        orientation_controls_layout.addWidget(self.normal_rotate_2_negative_button)

        self.thickness = QGroupBox('thickness')
        thickness_layout = QHBoxLayout(self.thickness)
        self.thickness_spinbox = QLabeledDoubleSlider(Qt.Horizontal, self)
        thickness_layout.addWidget(self.thickness_spinbox)
        self.thickness_spinbox.setMinimum(1.0)
        self.thickness_spinbox.setMaximum(64)
        self.thickness_spinbox.setValue(10)

        self.fun_buttons = QGroupBox('fun buttons')
        fun_layout = QVBoxLayout(self.fun_buttons)

        self.shift_normal_positive_button = QPushButton('shift along normal (+)')
        self.shift_normal_negative_button = QPushButton('shift along normal (-)')
        self.random_normal_button = QPushButton('random orientation', self)
        self.random_position_button = QPushButton('random position', self)

        fun_layout.addWidget(self.random_normal_button)
        fun_layout.addWidget(self.random_position_button)
        fun_layout.addWidget(self.shift_normal_positive_button)
        fun_layout.addWidget(self.shift_normal_negative_button)

        master_layout.addWidget(self.position_sliders)
        master_layout.addWidget(self.orientation_presets)
        master_layout.addWidget(self.orientation_controls)
        master_layout.addWidget(self.thickness)
        master_layout.addWidget(self.fun_buttons)

    @property
    def plane_position(self):
        return self.position_0.value(), self.position_1.value(), self.position_2.value()


def update_plane_position(plane_position):
    viewer.layers['plane'].plane.position = plane_position
    viewer.layers['plane point'].data = np.array(plane_position).reshape((-1, 3))

    plane_normal_layer.data[:, 0, :] = plane_position
    plane_normal_layer.data = plane_normal_layer.data


def update_plane_normal(plane_normal):
    viewer.layers['plane'].plane.normal_vector = plane_normal
    viewer.layers['plane normal'].data[:, 1, :] = plane_normal
    # trigger data change event
    viewer.layers['plane normal'].data = plane_normal_layer.data


def update_plane_thickness(widget):
    plane_layer.plane.thickness = widget.thickness_spinbox.value()


def rotate_plane(axis: int, positive: bool, angle: float = 1.5):
    current_normal = np.array(viewer.layers['plane'].plane.normal_vector).reshape(-1, 3,
                                                                                  1)
    normalised = current_normal / np.linalg.norm(current_normal)

    axis_str = 'xyz'[axis]
    if not positive:
        angle = -angle

    new_normal = R.from_euler(axis_str, [angle], degrees=True).as_matrix() @ normalised
    update_plane_normal(new_normal.reshape(-1))


def shift_plane_along_normal(positive: bool = True):
    plane_parameters = viewer.layers['plane'].plane
    position = np.array(plane_parameters.position, dtype=float)
    normal = np.array(plane_parameters.normal_vector, dtype=float)
    shift = normal / np.linalg.norm(normal)

    if not positive:
        shift = -shift

    new_position = position + shift
    update_plane_position(new_position)


def create_plane_widget():
    widget = PlaneWidget()

    widget.position_0.valueChanged.connect(lambda: update_plane_position(widget.plane_position))
    widget.position_1.valueChanged.connect(lambda: update_plane_position(widget.plane_position))
    widget.position_2.valueChanged.connect(lambda: update_plane_position(widget.plane_position))

    widget.normal_0_button.clicked.connect(lambda: update_plane_normal((1, 0, 0)))
    widget.normal_1_button.clicked.connect(lambda: update_plane_normal((0, 1, 0)))
    widget.normal_2_button.clicked.connect(lambda: update_plane_normal((0, 0, 1)))
    widget.normal_oblique_button.clicked.connect(lambda: update_plane_normal((1, 1, 1)))

    widget.normal_rotate_0_positive_button.clicked.connect(lambda: rotate_plane(0, positive=True))
    widget.normal_rotate_0_negative_button.clicked.connect(lambda: rotate_plane(0, positive=False))
    widget.normal_rotate_1_positive_button.clicked.connect(lambda: rotate_plane(1, positive=True))
    widget.normal_rotate_1_negative_button.clicked.connect(lambda: rotate_plane(1, positive=False))
    widget.normal_rotate_2_positive_button.clicked.connect(lambda: rotate_plane(2, positive=True))
    widget.normal_rotate_2_negative_button.clicked.connect(lambda: rotate_plane(2, positive=False))

    widget.random_normal_button.clicked.connect(
        lambda: update_plane_normal(np.random.uniform(low=0, high=1, size=3)))
    widget.random_position_button.clicked.connect(
        lambda: update_plane_position(np.random.uniform(low=0, high=64, size=3)))
    widget.shift_normal_positive_button.clicked.connect(
        lambda: shift_plane_along_normal(positive=True))
    widget.shift_normal_negative_button.clicked.connect(
        lambda: shift_plane_along_normal(positive=False))

    widget.thickness_spinbox.valueChanged.connect(
        lambda: update_plane_thickness(widget))
    return widget


plane_widget = create_plane_widget()
viewer.window.add_dock_widget(plane_widget)
napari.run()
