from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ...utils.geometry import (
    clamp_point_to_bounding_box,
    point_in_bounding_box,
)

if TYPE_CHECKING:
    from ...utils.events import Event
    from .image import Image


def move_plane_along_normal(layer: Image, event: Event):
    """Move a layers slicing plane along its normal vector on click and drag."""
    # early exit clauses
    if (
        'Shift' not in event.modifiers
        or layer.visible is False
        or layer.interactive is False
        or layer.experimental_slicing_plane.draggable is False
        or len(event.dims_displayed) < 3
    ):
        return

    # Store mouse position at start of drag
    initial_position = np.asarray(event.position)
    initial_view_direction = np.asarray(event.view_direction)

    # Calculate intersection of click with plane through data in data coordinates
    intersection = layer.experimental_slicing_plane.intersect_with_line(
        line_position=initial_position[event.dims_displayed],
        line_direction=initial_view_direction[event.dims_displayed],
    )

    # Check if click was on plane and if not, exit early.
    if not point_in_bounding_box(
        intersection, layer.extent.data[:, event.dims_displayed]
    ):
        return

    # Store original plane position and disable interactivity during plane drag
    original_plane_position = np.copy(
        layer.experimental_slicing_plane.position
    )
    layer.interactive = False

    yield

    while event.type == 'mouse_move':
        # Project mouse drag onto plane normal
        drag_distance = layer.projected_distance_from_mouse_drag(
            start_position=initial_position,
            end_position=np.asarray(event.position),
            view_direction=np.asarray(event.view_direction),
            vector=layer.experimental_slicing_plane.normal,
            dims_displayed=event.dims_displayed,
        )

        # Calculate updated plane position
        updated_position = original_plane_position + (
            drag_distance * np.array(layer.experimental_slicing_plane.normal)
        )

        clamped_plane_position = clamp_point_to_bounding_box(
            updated_position, layer._display_bounding_box(event.dims_displayed)
        )

        layer.experimental_slicing_plane.position = clamped_plane_position
        yield

    # Re-enable volume_layer interactivity after the drag
    layer.interactive = True
