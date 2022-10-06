"""
Clipping planes interactive
===========================

Display a 3D image (plus labels) with a clipping plane and interactive controls
for moving the plane

.. tags:: experimental
"""
import napari
import numpy as np
from skimage import data
from scipy import ndimage
from vispy.geometry import create_sphere

viewer = napari.Viewer(ndisplay=3)

# VOLUME and LABELS
blobs = data.binary_blobs(
    length=64, volume_fraction=0.1, n_dim=3
).astype(float)

labeled = ndimage.label(blobs)[0]

plane_parameters = {
    'position': (32, 32, 32),
    'normal': (1, 1, 1),
    'enabled': True
}

volume_layer = viewer.add_image(
    blobs, rendering='mip', name='volume',
    experimental_clipping_planes=[plane_parameters],
)

labels_layer = viewer.add_labels(
    labeled, name='labels', blending='translucent',
    experimental_clipping_planes=[plane_parameters],
)

# POINTS
points_layer = viewer.add_points(
    np.random.rand(20, 3) * 64, size=5,
    experimental_clipping_planes=[plane_parameters],
)

# SPHERE
mesh = create_sphere(method='ico')
sphere_vert = mesh.get_vertices() * 20
sphere_vert += 32
surface_layer = viewer.add_surface(
    (sphere_vert, mesh.get_faces()),
    experimental_clipping_planes=[plane_parameters],
)

# SHAPES
shapes_data = np.random.rand(3, 4, 3) * 64

shapes_layer = viewer.add_shapes(
    shapes_data,
    face_color=['magenta', 'green', 'blue'],
    experimental_clipping_planes=[plane_parameters],
)

# VECTORS
vectors = np.zeros((20, 2, 3))
vectors[:, 0] = 32
vectors[:, 1] = (np.random.rand(20, 3) - 0.5) * 32

vectors_layer = viewer.add_vectors(
    vectors,
    experimental_clipping_planes=[plane_parameters],
)


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
    volume_layer = viewer.layers['volume']

    # Calculate intersection of click with data bounding box
    near_point, far_point = volume_layer.get_ray_intersections(
        event.position,
        event.view_direction,
        event.dims_displayed,
    )

    # Calculate intersection of click with plane through data
    intersection = volume_layer.experimental_clipping_planes[0].intersect_with_line(
        line_position=near_point, line_direction=event.view_direction
    )

    # Check if click was on plane by checking if intersection occurs within
    # data bounding box. If so, exit early.
    if not point_in_bounding_box(intersection, volume_layer.extent.data):
        return

    # Get plane parameters in vispy coordinates (zyx -> xyz)
    plane_normal_data_vispy = np.array(volume_layer.experimental_clipping_planes[0].normal)[[2, 1, 0]]
    plane_position_data_vispy = np.array(volume_layer.experimental_clipping_planes[0].position)[[2, 1, 0]]

    # Get transform which maps from data (vispy) to canvas
    # note that we're using a private attribute here, which may not be present in future napari versions
    visual2canvas = viewer.window._qt_viewer.layer_to_visual[volume_layer].node.get_transform(
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
    volume_layer.interactive = False
    labels_layer.interactive = False
    labels_layer.interactive = False
    points_layer.interactive = False
    surface_layer.interactive = False
    shapes_layer.interactive = False
    vectors_layer.interactive = False

    # Store original plane position and start position in canvas coordinates
    original_plane_position = volume_layer.experimental_clipping_planes[0].position
    start_position_canv = event.pos

    yield
    while event.type == "mouse_move":
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
            volume_layer.experimental_clipping_planes[0].normal)

        if point_in_bounding_box(updated_position, volume_layer.extent.data):
            volume_layer.experimental_clipping_planes[0].position = updated_position
            labels_layer.experimental_clipping_planes[0].position = updated_position
            points_layer.experimental_clipping_planes[0].position = updated_position
            surface_layer.experimental_clipping_planes[0].position = updated_position
            shapes_layer.experimental_clipping_planes[0].position = updated_position
            vectors_layer.experimental_clipping_planes[0].position = updated_position

        yield

    # Re-enable
    volume_layer.interactive = True
    labels_layer.interactive = True
    points_layer.interactive = True
    surface_layer.interactive = True
    shapes_layer.interactive = True
    vectors_layer.interactive = True


viewer.axes.visible = True
viewer.camera.angles = (45, 45, 45)
viewer.camera.zoom = 5
viewer.text_overlay.update(dict(
    text='Drag the clipping plane surface to move it along its normal.',
    font_size=20,
    visible=True,
))

if __name__ == '__main__':
    napari.run()
