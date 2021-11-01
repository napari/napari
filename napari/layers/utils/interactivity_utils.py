from typing import List, Tuple, Union

import numpy as np

from ...utils.geometry import project_points_onto_plane


def click_plane_from_intersection_points(
    start_point: np.ndarray,
    end_point: np.ndarray,
    dims_displayed: Union[List[int], np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """Get the plane that is at the near side of the data bounding box and is
    normal to the click direction from the start_point and end_point of the
    click ray that intersects the data bounding box.

    Parameters
    ----------
    start_point : np.ndarray
        The first intersection of the click ray with the data bounding box.
    end_point : np.ndarray
        The second intersection of the click ray with the data bounding box.
    dims_displayed : Union[List[int], np.ndarray]
        The dimensions of the data array currently in view.

    Returns
    -------
    plane_point : np.ndarray
        The point on the plane that intersects the click ray. This is returned
        in data coordinates with only the dimensions that are displayed.
    plane_normal : np.ndarray
        The normal unit vector for the plane. It points in the direction of the click
        in data coordinates.
    """
    plane_point = start_point[dims_displayed]
    end_position_view = end_point[dims_displayed]
    ray_direction = end_position_view - plane_point
    plane_normal = ray_direction / np.linalg.norm(ray_direction)
    return plane_point, plane_normal


# get click plane from mouse event - tick

# project points on to click plane - tick

# rotate points and plane to be axis aligned - tick

# 2D intersection of click with points

# calculate signed distancdistancee between points and plane

# probably need to override get_value on the points layer to use a
# larger bounding box which fully encompasses all points


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
    end_position_canvas = project_points_onto_plane(
        end_position, start_position, view_direction
    )
    # Calculate the drag vector on the pseudo-canvas.
    drag_vector_canvas = np.squeeze(end_position_canvas - start_position)

    # Project the drag vector onto the specified vector(s), return the distance
    return np.einsum('j, ij -> i', drag_vector_canvas, vector).squeeze()
