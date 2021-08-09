"""
Display one 3-D volume layer using the add_volume API and display it as a plane
with a simple widget for modifying plane parameters
"""
import napari
import numpy as np
from napari._qt.qt_event_loop import get_app
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
    plane=plane_parameters
)

# add a points layer for interactively adding points
points_layer = viewer.add_points(name='points', ndim=3, face_color='cornflowerblue')


# Widget to control some plane params
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
    viewer.layers['plane'].plane.position = plane_position


def update_plane_thickness(widget):
    plane_layer.plane.thickness = widget.thickness_spinbox.value()


def create_plane_widget():
    widget = PlaneWidget()
    widget.position_slider.valueChanged.connect(
        lambda: update_plane_y_position(widget)
    )
    widget.thickness_spinbox.valueChanged.connect(
        lambda: update_plane_thickness(widget)
    )
    return widget


def point_in_bounding_box(point, bbox):
    if np.all(point > bbox[0]) and np.all(point < bbox[1]):
        return True
    return False


@viewer.mouse_drag_callbacks.append
def on_click(viewer, event):
    # get layers from viewer
    plane_layer = viewer.layers['plane']
    points_layer = viewer.layers['points']

    dragged = False

    # Calculate intersection of click with data bounding box
    near_point, far_point = plane_layer.get_ray_intersections(
        event.position,
        event.view_direction,
        event.dims_displayed,
    )

    # Calculate intersection of click with plane through data
    intersection = plane_layer.plane.intersect_with_line(
        line_position=near_point, line_direction=event.view_direction
    )

    # Check if intersection is within data bbox (click was on plane)
    # early exit, avoids excessive calculation
    if not point_in_bounding_box(intersection, plane_layer.extent.data):
        return

    # Get plane parameters in vispy coordinates (zyx -> xyz)
    plane_normal_data_vispy = np.array(plane_layer.plane.normal)[[2, 1, 0]]
    plane_position_data_vispy = np.array(plane_layer.plane.position)[[2, 1, 0]]

    # Get transform which maps from data (vispy) to canvas
    visual2canvas = viewer.window.qt_viewer.layer_to_visual[plane_layer].node.get_transform(
        map_from="visual", map_to="canvas"
    )

    # Find start and end positions of plane normal in canvas coordinates
    plane_normal_start_canvas = visual2canvas.map(plane_position_data_vispy)
    plane_normal_end_canvas = visual2canvas.map(plane_position_data_vispy + plane_normal_data_vispy)

    # Calculate plane normal vector in canvas coordinates
    plane_normal_canv = (plane_normal_end_canvas - plane_normal_start_canvas)[[0, 1]]
    plane_normal_canv_normalised = (
            plane_normal_canv / np.linalg.norm(plane_normal_canv)
    )

    # Disable interactivity during plane drag
    plane_layer.interactive = False

    # Store original plane position and start position in canvas coordinates
    original_plane_position = plane_layer.plane.position
    start_position_canv = event.pos

    yield
    while event.type == "mouse_move":
        # Set drag state to differentiate drag events from click events
        dragged = True

        # Get end position in canvas coordinates
        end_position_canv = event.pos

        # Calculate drag vector in canvas coordinates
        drag_vector_canv = end_position_canv - start_position_canv

        # Project the drag vector onto the plane normal vector
        # (in canvas coorinates)
        drag_projection_on_plane_normal = np.dot(
            drag_vector_canv, plane_normal_canv_normalised
        )

        # Update position of plane according to drag vector
        # only update if plane position is within data bounding box
        scale_factor = drag_projection_on_plane_normal / np.linalg.norm(plane_normal_canv)
        updated_position = original_plane_position + scale_factor * np.array(plane_layer.plane.normal)

        if point_in_bounding_box(updated_position, plane_layer.extent.data):
            plane_layer.plane.position = updated_position

        yield
    if dragged:
        pass
    else: # event was a click without a drag
        if point_in_bounding_box(intersection, plane_layer.extent.data):
            points_layer.add(intersection)
    plane_layer.interactive = True

app = get_app()
plane_widget = create_plane_widget()
viewer.window.add_dock_widget(
    plane_widget, name='Plane Widget', area='left'
)
viewer.axes.visible = True
viewer.camera.angles = (45, 45, 45)
viewer.camera.zoom = 5
napari.run()
