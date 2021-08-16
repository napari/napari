from dataclasses import dataclass

import numpy as np

from ...utils.geometry import point_in_bounding_box
from ..utils.interactivity_utils import mouse_events_to_projected_distance


def on_plane_drag(layer, event):
    """Shift a rendered plane along its normal vector.

    This function will shift a plane along its normal vector when the plane is
    clicked and dragged."""
    # Early exit if plane rendering not enabled or layer isn't visible
    if not (layer.embedded_plane.enabled and layer.visible):
        return

    # Calculate intersection of click with data bounding box in data coordinates
    near_point, far_point = layer.get_ray_intersections(
        event.position,
        event.view_direction,
        event.dims_displayed,
    )

    # exit if click is outside data bounding box
    if near_point is None and far_point is None:
        return

    # Calculate intersection of click with plane through data in data coordinates
    intersection = layer.embedded_plane.intersect_with_line(
        line_position=near_point, line_direction=event.view_direction
    )

    # Check if click was on plane by checking if intersection occurs within
    # data bounding box. If not, exit early.
    if not point_in_bounding_box(intersection, layer.extent.data):
        return

    # Store original plane position and disable interactivity during plane drag
    original_plane_position = layer.embedded_plane.position
    layer.interactive = False

    # store the whole event from the beginning of a drag
    # can't copy/serialise mouse events... event ref gets updated
    # temp solution, store necessary info in a simple dataclass
    @dataclass
    class FakeMouseEvent:
        position: tuple
        view_direction: tuple

    start_event = FakeMouseEvent(
        position=event.position, view_direction=event.view_direction
    )
    yield

    while event.type == 'mouse_move':
        drag_event = event

        # Project mouse drag onto plane normal
        drag_distance = mouse_events_to_projected_distance(
            start_event=start_event,
            end_event=drag_event,
            layer=layer,
            vector=layer.embedded_plane.normal,
        )

        layer.embedded_plane.position = (
            original_plane_position
            + drag_distance * np.array(layer.embedded_plane.normal)
        )
        # TODO clamp position to data bbox
        yield

    # Re-enable layer interactivity after the drag
    layer.interactive = True
