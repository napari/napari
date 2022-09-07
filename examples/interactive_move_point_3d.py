"""
Interactive move point
======================

3D click and drag interactivity demo

.. tags:: experimental
"""
from copy import copy

import numpy as np

import napari
from napari.utils.geometry import project_points_onto_plane

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
    name='bounding box',
    face_color='green',
    size=0.2,
    edge_width=0
)
point_layer = viewer.add_points(
    [0, 0, 0],
    name='point',
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

    # store start position of point
    original_position = copy(point_layer.data[0])

    yield
    while event.type == 'mouse_move':
        # Calculate click position in data coords
        point_to_project = np.asarray(layer.world_to_data(event.position))[
            list(event.dims_displayed)
        ]

        # Calculate view direction in data coordinates
        # this view direction, together with the click position, form a plane
        # parallel to the canvas in data coordinates.
        view_direction_data = np.asarray(layer._world_to_data_ray(
            list(event.view_direction)
        ))[event.dims_displayed]

        # Project click position onto plane
        projected_position = project_point_onto_plane(
            point=point_to_project,
            plane_point=original_position,
            plane_normal=view_direction_data,
        )

        # Calculate shifts to apply to point
        shifts = projected_position - original_position

        # Update position
        updated_position = original_position + shifts

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

if __name__ == '__main__':
    napari.run()
