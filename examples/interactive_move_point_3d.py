"""
3D click and drag interactivity demo
"""
from copy import copy

import numpy as np

import napari

# Create viewer, point to move and bounding box
viewer = napari.Viewer(ndisplay=3)

bounding_box_data = [
    [-1, -1, -1],
    [-1, -1, 1],
    [-1, 1, -1],
    [-1, 1, 1],
    [1, -1, -1],
    [1, -1, 1],
    [1, 1, -1],
    [1, 1, 1]
]
bounding_box_layer = viewer.add_points(
    bounding_box_data,
    face_color='green',
    size=0.2,
    edge_width=0
)
point_layer = viewer.add_points(
    [0, 0, 0],
    face_color='magenta',
    size=0.2,
    edge_width=0
)


@point_layer.mouse_drag_callbacks.append
def drag_along_camera_plane(layer, event):
    # early exit if shift isn't held
    if not 'Shift' in event.modifiers:
        return

    # disable interactivity during this drag event to fix the view direction
    layer.interactive = False

    # store start position of point and mouse event data
    original_position = copy(point_layer.data[0])
    start_position_world = copy(event.position)
    view_direction = copy(event.view_direction)

    # Calculate two orthogonal unit vectors in camera plane (in data
    # coordinates)
    # These provide a basis for moving our point
    arbitrary_vector = np.array([1, 1, 1])
    v1 = np.cross(view_direction, arbitrary_vector)  # orthogonal to both
    v2 = np.cross(view_direction, v1)  # orthogonal to both

    basis_vectors = np.stack([v1, v2], axis=0)
    basis_vectors = basis_vectors / np.linalg.norm(basis_vectors, axis=1)[:,
                                    np.newaxis]

    yield
    while event.type == 'mouse_move':
        # project drag vector onto orthogonal basis_vectors in pseudo-canvas
        current_position_world = event.position
        projected_distances = layer.projected_distance_from_mouse_drag(
            start_position_world, current_position_world, view_direction,
            basis_vectors
        )

        # Calculate shifts as projected distances multiplied by basis_vectors
        shifts = projected_distances[:, np.newaxis] * basis_vectors

        # Update position
        updated_position = original_position + np.sum(shifts, axis=0)

        # Clamp updated position to bounding box
        clamped = np.where(updated_position > 1, 1, updated_position)
        clamped = np.where(clamped < -1, -1, clamped)

        # update
        point_layer.data = clamped
        yield
    # reenable interactivity
    layer.interactive = True

# setup viewer
viewer.camera.angles = (45, 30, 30)
viewer.camera.zoom = 100
viewer.text_overlay.visible = True
viewer.text_overlay.text = """'shift' + click and drag to move the pink point
normal click and drag to rotate the scene
"""
napari.run()
