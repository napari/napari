"""
3D click and drag interactivity demo
"""
from dataclasses import dataclass

import numpy as np

import napari

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
bounding_box = viewer.add_points(bounding_box_data, face_color='green',
                                 size=0.2, edge_width=0)
point = viewer.add_points([0, 0, 0], face_color='magenta', size=0.2,
                          edge_width=0)


@point.mouse_drag_callbacks.append
def drag_along_camera_plane(layer, event):
    # early exit if shift isn't held
    if not 'Shift' in event.modifiers:
        return

    # disable interactivity
    layer.interactive = False

    # store start event
    @dataclass
    class FakeMouseEvent:
        position: tuple
        view_direction: tuple

    start_event = FakeMouseEvent(
        position=event.position, view_direction=event.view_direction
    )
    # Calculate two orthogonal vectors in camera plane (in data coordinates)
    # these are our basis vectors for calculating how to shift the point
    arbitrary_vector = np.array([1, 1, 1])
    v1 = np.cross(event.view_direction, arbitrary_vector)
    v2 = np.cross(event.view_direction, v1)

    basis_vectors = np.stack([v1, v2], axis=0)
    basis_vectors = basis_vectors / np.linalg.norm(basis_vectors, axis=1)[:,
                                    np.newaxis]

    original_position = point.data[0]
    yield
    while event.type == 'mouse_move':
        # project drag vector onto orthogonal basis_vectors in pseudo-canvas
        projected_distances = layer.projected_distance_from_mouse_events(
            start_event, event, basis_vectors
        )

        # Calculate shifts as projected distances multiplied by basis basis_vectors
        shifts = projected_distances[:, np.newaxis] * basis_vectors
        updated_position = original_position + np.sum(shifts, axis=0)

        # Clamp to bounding box
        clamped = np.where(updated_position > 1, 1, updated_position)
        clamped = np.where(clamped < -1, -1, clamped)
        point.data = clamped
        yield
    layer.interactive = True


viewer.camera.angles = (45, 30, 30)
viewer.camera.zoom = 100
viewer.text_overlay.visible = True
viewer.text_overlay.text = """'shift' + click and drag to move the pink point
click and drag to rotate the scene
"""
napari.run()
