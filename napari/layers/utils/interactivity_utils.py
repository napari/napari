from typing import List, Tuple, Union

import numpy as np


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
