"""
Display one 3-D volume layer using the add_volume API and display it as a plane
with interactive controls for moving the plane and adding points
"""
import napari
import numpy as np
from skimage import data

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

# add a points layer for interactively adding points
points_layer = viewer.add_points(name='points', ndim=3, face_color='cornflowerblue')


def point_in_bounding_box(point, bounding_box):
    if np.all(point > bounding_box[0]) and np.all(point < bounding_box[1]):
        return True
    return False


@viewer.mouse_drag_callbacks.append
def shift_plane_along_normal(viewer, event):
    """Shift a plane along its normal vector on mouse drag.

    This callback will shift a plane along its normal vector when the plane is
    clicked and dragged. The general strategy is to
    1) find both the plane normal vector and the mouse drag vector in canvas
    coordinates
    2) calculate how far to move the plane in canvas coordinates, this is done
    by projecting the mouse drag vector onto the (normalised) plane normal
    vector
    3) transform this drag distance (canvas coordinates) into data coordinates
    4) update the plane position

    It will also add a point to the points layer for a 'click-not-drag' event.
    """
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
    intersection = plane_layer.experimental_slicing_plane.intersect_with_line(
        line_position=near_point, line_direction=event.view_direction
    )

    # Check if click was on plane by checking if intersection occurs within
    # data bounding box. If so, exit early.
    if not point_in_bounding_box(intersection, plane_layer.extent.data):
        return

    # Get plane parameters in vispy coordinates (zyx -> xyz)
    plane_normal_data_vispy = np.array(plane_layer.experimental_slicing_plane.normal)[[2, 1, 0]]
    plane_position_data_vispy = np.array(plane_layer.experimental_slicing_plane.position)[[2, 1, 0]]

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
    original_plane_position = plane_layer.experimental_slicing_plane.position
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
        drag_distance_data = drag_projection_on_plane_normal / np.linalg.norm(plane_normal_canv)
        updated_position = original_plane_position + drag_distance_data * np.array(
            plane_layer.experimental_slicing_plane.normal)

        if point_in_bounding_box(updated_position, plane_layer.extent.data):
            plane_layer.experimental_slicing_plane.position = updated_position

        yield
    if dragged:
        pass
    else:  # event was a click without a drag
        if point_in_bounding_box(intersection, plane_layer.extent.data):
            points_layer.add(intersection)

    # Re-enable
    plane_layer.interactive = True


viewer.axes.visible = True
viewer.camera.angles = (45, 45, 45)
viewer.camera.zoom = 5
napari.run()
