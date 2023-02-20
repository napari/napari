from typing import Tuple, Union

import numpy as np

from napari.layers.utils._text_constants import Anchor
from napari.utils.translations import trans


def get_text_anchors(
    view_data: Union[np.ndarray, list],
    ndisplay: int,
    anchor: Anchor = Anchor.CENTER,
) -> np.ndarray:
    # Explicitly convert to an Anchor so that string values can be used.
    text_anchor_func = TEXT_ANCHOR_CALCULATION[Anchor(anchor)]
    text_coords, anchor_x, anchor_y = text_anchor_func(view_data, ndisplay)
    return text_coords, anchor_x, anchor_y


def _calculate_anchor_center(
    view_data: Union[np.ndarray, list], ndisplay: int
) -> Tuple[np.ndarray, str, str]:
    text_coords = _calculate_bbox_centers(view_data)

    anchor_x = 'center'
    anchor_y = 'center'

    return text_coords, anchor_x, anchor_y


def _calculate_bbox_centers(view_data: Union[np.ndarray, list]) -> np.ndarray:
    if isinstance(view_data, np.ndarray):
        if view_data.ndim == 2:
            # if the data are a list of coordinates, just return the coord (e.g., points)
            bbox_centers = view_data
        else:
            bbox_centers = np.mean(view_data, axis=0)
    elif isinstance(view_data, list):
        bbox_centers = np.array(
            [np.mean(coords, axis=0) for coords in view_data]
        )
    else:
        raise TypeError(
            trans._(
                'view_data should be a numpy array or list when using Anchor.CENTER',
                deferred=True,
            )
        )
    return bbox_centers


def _calculate_anchor_upper_left(
    view_data: Union[np.ndarray, list], ndisplay: int
) -> Tuple[np.ndarray, str, str]:
    if ndisplay == 2:
        bbox_min, bbox_max = _calculate_bbox_extents(view_data)
        text_anchors = np.array([bbox_min[:, 0], bbox_min[:, 1]]).T

        anchor_x = 'left'
        anchor_y = 'top'
    else:
        # in 3D, use centered anchor
        text_anchors, anchor_x, anchor_y = _calculate_anchor_center(
            view_data, ndisplay
        )

    return text_anchors, anchor_x, anchor_y


def _calculate_anchor_upper_right(
    view_data: Union[np.ndarray, list], ndisplay: int
) -> Tuple[np.ndarray, str, str]:
    if ndisplay == 2:
        bbox_min, bbox_max = _calculate_bbox_extents(view_data)
        text_anchors = np.array([bbox_min[:, 0], bbox_max[:, 1]]).T

        anchor_x = 'right'
        anchor_y = 'top'
    else:
        # in 3D, use centered anchor
        text_anchors, anchor_x, anchor_y = _calculate_anchor_center(
            view_data, ndisplay
        )

    return text_anchors, anchor_x, anchor_y


def _calculate_anchor_lower_left(
    view_data: Union[np.ndarray, list], ndisplay: int
) -> Tuple[np.ndarray, str, str]:
    if ndisplay == 2:
        bbox_min, bbox_max = _calculate_bbox_extents(view_data)
        text_anchors = np.array([bbox_max[:, 0], bbox_min[:, 1]]).T

        anchor_x = 'left'
        anchor_y = 'bottom'
    else:
        # in 3D, use centered anchor
        text_anchors, anchor_x, anchor_y = _calculate_anchor_center(
            view_data, ndisplay
        )

    return text_anchors, anchor_x, anchor_y


def _calculate_anchor_lower_right(
    view_data: Union[np.ndarray, list], ndisplay: int
) -> Tuple[np.ndarray, str, str]:
    if ndisplay == 2:
        bbox_min, bbox_max = _calculate_bbox_extents(view_data)
        text_anchors = np.array([bbox_max[:, 0], bbox_max[:, 1]]).T

        anchor_x = 'right'
        anchor_y = 'bottom'
    else:
        # in 3D, use centered anchor
        text_anchors, anchor_x, anchor_y = _calculate_anchor_center(
            view_data, ndisplay
        )

    return text_anchors, anchor_x, anchor_y


def _calculate_bbox_extents(view_data: Union[np.ndarray, list]) -> np.ndarray:
    """Calculate the extents of the bounding box"""
    if isinstance(view_data, np.ndarray):
        if view_data.ndim == 2:
            # if the data are a list of coordinates, just return the coord (e.g., points)
            bbox_min = view_data
            bbox_max = view_data
        else:
            bbox_min = np.min(view_data, axis=0)
            bbox_max = np.max(view_data, axis=0)
    elif isinstance(view_data, list):
        bbox_min = np.array([np.min(coords, axis=0) for coords in view_data])
        bbox_max = np.array([np.max(coords, axis=0) for coords in view_data])
    else:
        raise TypeError(
            trans._(
                'view_data should be a numpy array or list',
                deferred=True,
            )
        )
    return bbox_min, bbox_max


TEXT_ANCHOR_CALCULATION = {
    Anchor.CENTER: _calculate_anchor_center,
    Anchor.UPPER_LEFT: _calculate_anchor_upper_left,
    Anchor.UPPER_RIGHT: _calculate_anchor_upper_right,
    Anchor.LOWER_LEFT: _calculate_anchor_lower_left,
    Anchor.LOWER_RIGHT: _calculate_anchor_lower_right,
}
