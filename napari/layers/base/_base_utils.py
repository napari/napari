import numpy as np

from ...utils.geometry import project_point_onto_plane


def mouse_events_to_projected_distance_data(
    start_event, end_event, layer, axis
):
    """Calculate the projected distance between two mouse events.

    The projection of the distance between two mouse events onto a 3D axis
    specified in data coordinates.

    Parameters
    ----------
    start_event
    end_event
    layer
    axis

    Returns
    -------
    projected_distance : float
    """
    # Store the start and end positions in world coordinates
    start_position = start_event.position[layer._dims_displayed_mask]
    end_position = end_event.position[layer._dims_displayed_mask]

    # Project the start and end positions onto a pseudo-canvas, a plane
    # parallel to the rendered canvas in data coordinates.
    start_position_canvas = start_position
    end_position_canvas = project_point_onto_plane(
        end_position, start_position_canvas, start_event.view_direction
    )
    # Calculate the drag vector on the pseudo-canvas.
    drag_vector_canvas = end_position_canvas - start_position_canvas

    # Project the drag vector onto the specified axis and return the distance.
    return np.dot(drag_vector_canvas, axis)
