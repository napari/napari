import re
from typing import Tuple, Union

import numpy as np

from ...utils.translations import trans
from ._text_constants import Anchor, TextMode


def get_text_anchors(
    view_data: Union[np.ndarray, list],
    ndisplay: int,
    anchor: Anchor = Anchor.CENTER,
) -> np.ndarray:
    text_anchor_func = TEXT_ANCHOR_CALCULATION[anchor]
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


def format_text_properties(text: str, n_text: int, properties: dict = {}):

    # If the text value is a property key, the text is the property values
    if text in properties:
        formatted_text = np.resize([str(v) for v in properties[text]], n_text)
        text_mode = TextMode.PROPERTY
    elif ('{' in text) and ('}' in text):
        format_keys = _get_format_keys(text, properties)
        formatted_text = _format_text_f_string(
            text=text,
            n_text=n_text,
            format_keys=format_keys,
            properties=properties,
        )
        text_mode = TextMode.FORMATTED

    else:
        formatted_text, text_mode = format_text_direct(text, n_text)

    return np.array(formatted_text), text_mode


def format_text_direct(text, n_text: int):
    if isinstance(text, str):
        formatted_text = np.repeat(text, n_text)
    else:
        if len(text) != n_text:
            raise ValueError(
                trans._(
                    'Number of text elements ({length}) should equal the length of the data ({n_text})',
                    deferred=True,
                    length=len(text),
                    n_text=n_text,
                )
            )

        formatted_text = np.asarray(text)

    text_mode = TextMode.NONE

    return formatted_text, text_mode


def _get_format_keys(text: str, properties: dict):
    format_keys = re.findall('{(.*?)}', text)

    format_keys_in_properties = []

    for format_key in format_keys:
        split_key = format_key.split(':')
        if split_key[0] in properties:
            if len(split_key) == 1:
                format_keys_in_properties.append((split_key[0], ''))
            else:
                format_keys_in_properties.append((split_key[0], split_key[1]))

    return format_keys_in_properties


def _format_text_f_string(
    text: str, n_text: int, format_keys: list, properties: dict
):

    all_formatted_text = []
    for i in range(n_text):
        formatted_text = text
        for format_key in format_keys:
            prop_value = properties[format_key[0]][i]
            string_template = '{' + ':' + format_key[1] + '}'
            formatted_prop_value = string_template.format(prop_value)
            if len(format_key[1]) == 0:
                original_value = '{' + format_key[0] + '}'
            else:
                original_value = (
                    '{' + format_key[0] + ':' + format_key[1] + '}'
                )
            formatted_text = formatted_text.replace(
                original_value, formatted_prop_value
            )
        all_formatted_text.append(formatted_text)

    return all_formatted_text
