"""
Interactive move rectangle
==========================

Shift a rectangle along its normal vector in 3D

.. tags:: experimental
"""

import numpy as np

import napari

rectangle = np.array(
    [
        [50, 75, 75],
        [50, 125, 75],
        [100, 125, 125],
        [100, 75, 125]
    ],
    dtype=float
)

shapes_data = np.array(rectangle)
normal_vector = np.cross(
    rectangle[0] - rectangle[1], rectangle[2] - rectangle[1]
)
normal_vector /= np.linalg.norm(normal_vector)

viewer = napari.Viewer(ndisplay=3)

shapes_layer = viewer.add_shapes(
    data=shapes_data,
    face_color='blue'
)
viewer.camera.angles = (-170, -20, -170)
viewer.camera.zoom = 1.5
viewer.text_overlay.visible = True
viewer.text_overlay.text = """'click and drag the rectangle to create copies along its normal vector
"""


@shapes_layer.mouse_drag_callbacks.append
def move_rectangle_along_normal(layer, event):
    shape_index, _ = layer.get_value(
        position=event.position,
        view_direction=event.view_direction,
        dims_displayed=event.dims_displayed
    )
    if shape_index is None:
        return

    layer.interactive = False

    start_position = np.copy(event.position)
    yield
    while event.type == 'mouse_move':
        projected_distance = layer.projected_distance_from_mouse_drag(
            start_position=start_position,
            end_position=event.position,
            view_direction=event.view_direction,
            vector=normal_vector,
            dims_displayed=event.dims_displayed,
        )
        shift_data_coordinates = projected_distance * normal_vector
        new_rectangle = layer.data[shape_index] + shift_data_coordinates
        layer.add(new_rectangle)
        yield
    layer.interactive = True


if __name__ == '__main__':
    napari.run()
