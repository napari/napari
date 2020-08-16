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
    _format_text_f_string,
    _get_format_keys,
    format_text_direct,
    format_text_properties,
    get_text_anchors,
)


def test_get_format_keys():
    string_with_keys = 'asdfa{hi}asd{hello:.2f}fasdf{yo}wind{sdfs'
    properties = {'hi': [1, 2, 3], 'hello': [1.1, 1.2, 1.3]}

    format_keys_in_properties = _get_format_keys(string_with_keys, properties)
    np.testing.assert_equal(
        format_keys_in_properties, [('hi', ''), ('hello', '.2f')]
    )

    string_without_valid_keys = 'asdfa{blah}asdfasdf{yo}wind{sdfs'
    format_keys_in_properties = _get_format_keys(
        string_without_valid_keys, properties
    )
    assert format_keys_in_properties == []

    string_without_keys = '3123123sdfsed'
    format_keys_in_properties = _get_format_keys(
        string_without_keys, properties
    )
    assert format_keys_in_properties == []


def test_format_text():
    text = 'asdfas{hi:.2f}adfsdf{hello}sdfshikskhello'
    n_text = 2
    format_keys = [
        ('hi', '.2f'),
        ('hello', ''),
    ]
    properties = {
        'hi': np.array([1.1234, 2.1]),
        'hello': np.array(['hola', 'bonjour']),
    }

    formatted_text = _format_text_f_string(
        text, n_text=n_text, format_keys=format_keys, properties=properties
    )
    expected_text = [
        'asdfas1.12adfsdfholasdfshikskhello',
        'asdfas2.10adfsdfbonjoursdfshikskhello',
    ]
    np.testing.assert_equal(formatted_text, expected_text)


def test_format_text_direct():
    text = 'hi'
    n_text = 5
    formatted_text, text_mode = format_text_direct(text, n_text=n_text)
    expected_text = np.repeat(text, n_text)
    np.testing.assert_equal(formatted_text, expected_text)

    text_as_list = n_text * ['hi']
    formatted_text, text_mode = format_text_direct(text_as_list, n_text=n_text)
    np.testing.assert_equal(formatted_text, expected_text)

    with pytest.raises(ValueError):
        wrong_text_length = ['hi', 'hi']
        formatted_text = format_text_direct(wrong_text_length, n_text=n_text)


def test_format_text_properties_no_matching_props():
    text = 'hi'
    n_text = 5
    formatted_text, text_mode = format_text_properties(text, n_text=5)
    expected_text = np.repeat(text, n_text)
    np.testing.assert_equal(formatted_text, expected_text)

    properties = {'hello': [1, 2, 3, 4, 5]}
    formatted_text, text_mode = format_text_properties(
        text, n_text=5, properties=properties
    )
    expected_text = np.repeat(text, n_text)
    np.testing.assert_equal(formatted_text, expected_text)


def test_format_text_properties():
    text = 'name'
    n_text = 2
    properties = {
        'name': np.array(['bob', 'jane']),
        'item': np.array(['ducks', 'microscopes']),
        'weight': np.array([3, 15.123]),
    }

    formatted_text, text_mode = format_text_properties(
        text, n_text, properties
    )
    expected_text = ['bob', 'jane']
    np.testing.assert_equal(formatted_text, expected_text)
    assert isinstance(formatted_text, np.ndarray)


def test_format_text_properties_f_string():
    text = '{name} has 5 {item} that weigh {weight:.1f} kgs'
    n_text = 2
    properties = {
        'name': np.array(['bob', 'jane']),
        'item': np.array(['ducks', 'microscopes']),
        'weight': np.array([3, 15.123]),
    }

    formatted_text, text_mode = format_text_properties(
        text, n_text, properties
    )
    expected_text = [
        'bob has 5 ducks that weigh 3.0 kgs',
        'jane has 5 microscopes that weigh 15.1 kgs',
    ]

    np.testing.assert_equal(formatted_text, expected_text)
    assert isinstance(formatted_text, np.ndarray)


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
