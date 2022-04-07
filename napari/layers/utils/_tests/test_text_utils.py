import numpy as np
import pytest

from napari.layers.utils._text_constants import Anchor
from napari.layers.utils._text_utils import (
    _calculate_anchor_center,
    _calculate_anchor_lower_left,
    _calculate_anchor_lower_right,
    _calculate_anchor_upper_left,
    _calculate_anchor_upper_right,
    _calculate_bbox_centers,
    _calculate_bbox_extents,
    get_text_anchors,
)

coords = np.array([[0, 0], [10, 0], [0, 10], [10, 10]])
view_data_list = [coords]
view_data_ndarray = coords


@pytest.mark.parametrize(
    "view_data,expected_coords",
    [(view_data_list, [[5, 5]]), (view_data_ndarray, coords)],
)
def test_bbox_center(view_data, expected_coords):
    """Unit test for _calculate_anchor_center. Roundtrip test in test_get_text_anchors"""
    anchor_data = _calculate_anchor_center(view_data, ndisplay=2)
    expected_anchor_data = (expected_coords, 'center', 'center')
    np.testing.assert_equal(anchor_data, expected_anchor_data)


@pytest.mark.parametrize(
    "view_data,expected_coords",
    [(view_data_list, [[0, 0]]), (view_data_ndarray, coords)],
)
def test_bbox_upper_left(view_data, expected_coords):
    """Unit test for _calculate_anchor_upper_left. Roundtrip test in test_get_text_anchors"""
    expected_anchor_data = (expected_coords, 'left', 'top')
    anchor_data = _calculate_anchor_upper_left(view_data, ndisplay=2)
    np.testing.assert_equal(anchor_data, expected_anchor_data)


@pytest.mark.parametrize(
    "view_data,expected_coords",
    [(view_data_list, [[0, 10]]), (view_data_ndarray, coords)],
)
def test_bbox_upper_right(view_data, expected_coords):
    """Unit test for _calculate_anchor_upper_right. Roundtrip test in test_get_text_anchors"""
    expected_anchor_data = (expected_coords, 'right', 'top')
    anchor_data = _calculate_anchor_upper_right(view_data, ndisplay=2)
    np.testing.assert_equal(anchor_data, expected_anchor_data)


@pytest.mark.parametrize(
    "view_data,expected_coords",
    [(view_data_list, [[10, 0]]), (view_data_ndarray, coords)],
)
def test_bbox_lower_left(view_data, expected_coords):
    """Unit test for _calculate_anchor_lower_left. Roundtrip test in test_get_text_anchors"""
    expected_anchor_data = (expected_coords, 'left', 'bottom')
    anchor_data = _calculate_anchor_lower_left(view_data, ndisplay=2)
    np.testing.assert_equal(anchor_data, expected_anchor_data)


@pytest.mark.parametrize(
    "view_data,expected_coords",
    [(view_data_list, [[10, 10]]), (view_data_ndarray, coords)],
)
def test_bbox_lower_right(view_data, expected_coords):
    """Unit test for _calculate_anchor_lower_right. Roundtrip test in test_get_text_anchors"""
    expected_anchor_data = (expected_coords, 'right', 'bottom')
    anchor_data = _calculate_anchor_lower_right(view_data, ndisplay=2)
    np.testing.assert_equal(anchor_data, expected_anchor_data)


@pytest.mark.parametrize(
    "anchor_type,ndisplay,expected_coords",
    [
        (Anchor.CENTER, 2, [[5, 5]]),
        (Anchor.UPPER_LEFT, 2, [[0, 0]]),
        (Anchor.UPPER_RIGHT, 2, [[0, 10]]),
        (Anchor.LOWER_LEFT, 2, [[10, 0]]),
        (Anchor.LOWER_RIGHT, 2, [[10, 10]]),
        (Anchor.CENTER, 3, [[5, 5]]),
        (Anchor.UPPER_LEFT, 3, [[5, 5]]),
        (Anchor.UPPER_RIGHT, 3, [[5, 5]]),
        (Anchor.LOWER_LEFT, 3, [[5, 5]]),
        (Anchor.LOWER_RIGHT, 3, [[5, 5]]),
    ],
)
def test_get_text_anchors(anchor_type, ndisplay, expected_coords):
    """Round trip tests for getting anchor coordinates."""
    coords = [np.array([[0, 0], [10, 0], [0, 10], [10, 10]])]
    anchor_coords, _, _ = get_text_anchors(
        coords, anchor=anchor_type, ndisplay=ndisplay
    )
    np.testing.assert_equal(anchor_coords, expected_coords)


def test_bbox_centers_exception():
    """_calculate_bbox_centers should raise a TypeError for non ndarray or list inputs"""
    with pytest.raises(TypeError):
        _ = _calculate_bbox_centers({'bad_data_type': True})


def test_bbox_extents_exception():
    """_calculate_bbox_extents should raise a TypeError for non ndarray or list inputs"""
    with pytest.raises(TypeError):
        _ = _calculate_bbox_extents({'bad_data_type': True})
