import numpy as np

from ...utils.geometry import point_in_bounding_box


def on_embedded_plane_drag(layer, event):
    """Shift a rendered plane along its normal vector.

    This function will shift a plane along its normal vector when the plane is
    clicked and dragged. The general strategy is to
    1) find both the plane normal vector and the mouse drag vector in canvas
    coordinates
    2) calculate how far to move the plane in canvas coordinates, this is done
    by projecting the mouse drag vector onto the (normalised) plane normal
    vector
    3) transform this drag distance (canvas coordinates) into data coordinates
    4) update the plane position
    """
    # local import to avoid circular import
    from ...viewer import current_viewer

    if not layer.embedded_plane.enabled:  # early exit clause
        return

    # Get a reference to the viewer and store current drag state
    viewer = current_viewer()
    dragged = False

    # Calculate intersection of click with data bounding box
    near_point, far_point = layer.get_ray_intersections(
        event.position,
        event.view_direction,
        event.dims_displayed,
    )

    # exit if click is outside data bounding box
    if near_point is None and far_point is None:
        return

    # Calculate intersection of click with plane through data
    intersection = layer.embedded_plane.intersect_with_line(
        line_position=near_point, line_direction=event.view_direction
    )

    # Check if click was on plane by checking if intersection occurs within
    # data bounding box. If so, exit early.
    if not point_in_bounding_box(intersection, layer.extent.data):
        return

    # Get transform which maps from data (vispy) to canvas
    visual2canvas = viewer.window.qt_viewer.layer_to_visual[
        layer
    ].node.get_transform(map_from="visual", map_to="canvas")

    # Get plane parameters in vispy coordinates (zyx -> xyz)
    plane_normal_data_vispy = np.array(layer.embedded_plane.normal)[[2, 1, 0]]
    plane_position_data_vispy = np.array(layer.embedded_plane.position)[
        [2, 1, 0]
    ]

    # Find start and end positions of plane normal in canvas coordinates
    plane_normal_start_canvas = visual2canvas.map(plane_position_data_vispy)
    plane_normal_end_canvas = visual2canvas.map(
        plane_position_data_vispy + plane_normal_data_vispy
    )

    # Calculate plane normal vector in canvas coordinates
    plane_normal_canv = (plane_normal_end_canvas - plane_normal_start_canvas)[
        [0, 1]
    ]
    plane_normal_canv_normalised = plane_normal_canv / np.linalg.norm(
        plane_normal_canv
    )

    # Disable interactivity during plane drag
    layer.interactive = False

    # Store original plane position and start position in canvas coordinates
    original_plane_position = layer.embedded_plane.position
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
        drag_distance_data = drag_projection_on_plane_normal / np.linalg.norm(
            plane_normal_canv
        )
        updated_position = (
            original_plane_position
            + drag_distance_data * np.array(layer.embedded_plane.normal)
        )

        if point_in_bounding_box(updated_position, layer.extent.data):
            layer.embedded_plane.position = updated_position

        yield
    if dragged:
        pass
    else:  # event was a click without a drag
        pass  # call a set of 'on plane click' callbacks?

    # Re-enable layer interactivity after the drag
    layer.interactive = True
