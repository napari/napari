import numpy as np

from napari.utils.geometry import project_point_onto_plane


def mouse_events_to_projected_distance(start_event, end_event, layer, vector):
    """Calculate the projected distance between two mouse events.
    Project the drag vector between two mouse events onto a 3D axis
    specified in data coordinates. The layer is used to get the positions
    in data coordinates from the events, where it is defined in world
    coordinates.
    The general strategy is to
    1) find mouse drag start and end positions, project them onto a
       pseudo-canvas (a plane aligned with the canvas) in data coordinates.
    2) project the mouse drag vector onto the (normalised) vector in data
       coordinates
    Parameters
    ----------
    start_event : Event
        Mouse event for the starting point
    end_event : Event
        Mouse event for the end point
    layer : Layer
        Layer in which the vector on which to project the drag vector is
        defined.
    vector : np.ndarray
        (3,) unit vector or (n, 3) array thereof on which to project the drag
        vector from start_event to end_event. This argument is defined in data
        coordinates.
    Returns
    -------
    projected_distance : float
    """
    # enforce at least 2d input
    vector = np.atleast_2d(vector)

    # Store the start and end positions in world coordinates
    start_position = np.array(start_event.position)[layer._dims_mask]
    end_position = np.array(end_event.position)[layer._dims_mask]

    # Project the start and end positions onto a pseudo-canvas, a plane
    # parallel to the rendered canvas in data coordinates.
    start_position_canvas = start_position
    end_position_canvas = project_point_onto_plane(
        end_position, start_position_canvas, start_event.view_direction
    )
    # Calculate the drag vector on the pseudo-canvas.
    drag_vector_canvas = np.squeeze(
        end_position_canvas - start_position_canvas
    )

    # Project the drag vector onto the specified axis and return the distance.
    return np.einsum('j, ij -> i', drag_vector_canvas, vector).squeeze()
