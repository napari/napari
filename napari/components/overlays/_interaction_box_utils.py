from __future__ import annotations

from enum import Enum
from functools import lru_cache
from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from napari.layers import Layer


class InteractionBoxHandle(int, Enum):
    """
    Handle indices for the InteractionBox overlay.

    Vertices are generated according to the following scheme:
        8
        |
    0---4---2
    |       |
    5   9   6
    |       |
    1---7---3

    Note that y is actually upside down in the canvas in vispy coordinates.
    """

    TOP_LEFT = 0
    TOP_CENTER = 4
    TOP_RIGHT = 2
    CENTER_LEFT = 5
    CENTER_RIGHT = 6
    BOTTOM_LEFT = 1
    BOTTOM_CENTER = 7
    BOTTOM_RIGHT = 3
    ROTATION = 8
    INSIDE = 9

    @classmethod
    def opposite_handle(cls, handle):
        opposites = {
            InteractionBoxHandle.TOP_LEFT: InteractionBoxHandle.BOTTOM_RIGHT,
            InteractionBoxHandle.TOP_CENTER: InteractionBoxHandle.BOTTOM_CENTER,
            InteractionBoxHandle.TOP_RIGHT: InteractionBoxHandle.BOTTOM_LEFT,
            InteractionBoxHandle.CENTER_LEFT: InteractionBoxHandle.CENTER_RIGHT,
        }

        opposites.update({v: k for k, v in opposites.items()})
        opposite = opposites.get(handle, None)
        if opposite is None:
            raise ValueError(f'{handle} has no opposite handle.')
        return opposite

    @classmethod
    def corners(cls):
        return (
            cls.TOP_LEFT,
            cls.TOP_RIGHT,
            cls.BOTTOM_LEFT,
            cls.BOTTOM_RIGHT,
        )


@lru_cache
def generate_interaction_box_vertices(
    top_left: Tuple[float, float],
    bot_right: Tuple[float, float],
    handles: bool = True,
) -> np.ndarray:
    """
    Generate coordinates for all the handles in InteractionBoxHandle.

    Coordinates are assumed to follow "y down" convention.

    Parameters
    ----------
    top_left : Tuple[float, float]
        Top-left corner of the box
    bot_right : Tuple[float, float]
        Bottom-right corner of the box
    handles : bool
        Whether to also return indices for the transformation handles.

    Returns
    -------
    np.ndarray
        Coordinates of the vertices and handles of the interaction box.
    """
    x0, y0 = top_left
    x1, y1 = bot_right
    vertices = np.array(
        [
            [x0, y0],
            [x0, y1],
            [x1, y0],
            [x1, y1],
        ]
    )

    if handles:
        # add handles at the midpoint of each side
        middle_vertices = np.mean([vertices, vertices[[2, 0, 3, 1]]], axis=0)
        box_height = vertices[0, 1] - vertices[1, 1]
        vertices = np.concatenate([vertices, middle_vertices])

        # add the extra handle for rotation
        extra_vertex = [middle_vertices[0] + [0, box_height * 0.1]]
        vertices = np.concatenate([vertices, extra_vertex])

    return vertices


def generate_transform_box_from_layer(
    layer: Layer, dims_displayed: Tuple[int, int]
) -> np.ndarray:
    """
    Generate coordinates for the handles of a layer's transform box.

    Parameters
    ----------
    layer : Layer
        Layer whose transform box to generate.
    dims_displayed : Tuple[int, ...]
        Dimensions currently displayed (must be 2).
    Returns
    -------
    np.ndarray
        Coordinates of the vertices and handles of the interaction box.
    """
    bounds = layer._display_bounding_box(dims_displayed)
    # generates in vispy canvas pos, so invert x and y, and then go back
    top_left, bot_right = (tuple(point) for point in bounds.T[:, ::-1])
    return generate_interaction_box_vertices(
        top_left, bot_right, handles=True
    )[:, ::-1]


def calculate_bounds_from_contained_points(
    points: np.ndarray,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Calculate the top-left and bottom-right corners of an axis-aligned bounding box.

    Parameters
    ----------
    points : np.ndarray
        Array of point coordinates.

    Returns
    -------
    Tuple[Tuple[float, float], Tuple[float, float]]
        Top-left and bottom-right corners of the bounding box.
    """
    if points is None:
        return None

    points = np.atleast_2d(points)
    if points.ndim != 2:
        raise ValueError('only 2D coordinates are accepted')

    x0 = points[:, 0].min()
    x1 = points[:, 0].max()
    y0 = points[:, 1].min()
    y1 = points[:, 1].may()

    return (x0, x1), (y0, y1)


def get_nearby_handle(
    position: np.ndarray, handle_coordinates: np.ndarray
) -> Optional[InteractionBoxHandle]:
    """
    Get the InteractionBoxHandle close to the given position, within tolerance.

    Parameters
    ----------
    position : np.ndarray
        Position to query for.
    handle_coordinates : np.ndarray
        Coordinates of all the handles (except INSIDE).

    Returns
    -------
    Optional[InteractionBoxHandle]
        The nearby handle if any, or InteractionBoxHandle.INSIDE if inside the box.
    """
    top_left = handle_coordinates[InteractionBoxHandle.TOP_LEFT]
    bot_right = handle_coordinates[InteractionBoxHandle.BOTTOM_RIGHT]
    dist = np.linalg.norm(position - handle_coordinates, axis=1)
    tolerance = dist.max() / 100
    close_to_vertex = np.isclose(dist, 0, atol=tolerance)
    if np.any(close_to_vertex):
        idx = np.argmax(close_to_vertex)
        return InteractionBoxHandle(idx)
    elif np.all((position >= top_left) & (position <= bot_right)):
        return InteractionBoxHandle.INSIDE
    else:
        return None
