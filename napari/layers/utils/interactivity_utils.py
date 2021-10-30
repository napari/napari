import numpy as np

from napari.utils.geometry import project_point_onto_plane


def drag_data_to_projected_distance(
    start_position, end_position, view_direction, vector
):
    """Calculate the projected distance between two mouse events.

    Project the drag vector between two mouse events onto a 3D vector
    specified in data coordinates.

    The general strategy is to
    1) find mouse drag start and end positions, project them onto a
       pseudo-canvas (a plane aligned with the canvas) in data coordinates.
    2) project the mouse drag vector onto the (normalised) vector in data
       coordinates
    Parameters
    ----------
    start_position : np.ndarray
        Starting point of the drag vector in data coordinates
    end_position : np.ndarray
        End point of the drag vector in data coordinates
    view_direction : np.ndarray
        Vector defining the plane normal of the plane onto which the drag
        vector is projected.
    vector : np.ndarray
        (3,) unit vector or (n, 3) array thereof on which to project the drag
        vector from start_event to end_event. This argument is defined in data
        coordinates.
    Returns
    -------
    projected_distance : (1, ) or (n, ) np.ndarray of float
    """
    # enforce at least 2d input
    vector = np.atleast_2d(vector)

    # Store the start and end positions in world coordinates
    start_position = np.asarray(start_position)
    end_position = np.asarray(end_position)

    # Project the start and end positions onto a pseudo-canvas, a plane
    # parallel to the rendered canvas in data coordinates.
    end_position_canvas = project_point_onto_plane(
        end_position, start_position, view_direction
    )
    # Calculate the drag vector on the pseudo-canvas.
    drag_vector_canvas = np.squeeze(end_position_canvas - start_position)

    # Project the drag vector onto the specified vector(s), return the distance
    return np.einsum('j, ij -> i', drag_vector_canvas, vector).squeeze()
