import numpy as np
import pytest

from napari.layers.utils.text_utils import (
    _get_format_keys,
    _format_text_f_string,
    format_text_properties,
    format_text_direct,
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
