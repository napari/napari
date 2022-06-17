import numpy as np

from napari.layers.utils.color_manager_utils import (
    guess_continuous,
    is_color_mapped,
)


def test_guess_continuous():
    continuous_annotation = np.array([1, 2, 3], dtype=np.float32)
    assert guess_continuous(continuous_annotation)

    categorical_annotation_1 = np.array([True, False], dtype=bool)
    assert not guess_continuous(categorical_annotation_1)

    categorical_annotation_2 = np.array([1, 2, 3], dtype=int)
    assert not guess_continuous(categorical_annotation_2)


def test_is_colormapped_string():
    color = 'hello'
    properties = {
        'hello': np.array([1, 1, 1, 1]),
        'hi': np.array([1, 0, 0, 1]),
    }
    assert is_color_mapped(color, properties)
    assert not is_color_mapped('red', properties)


def test_is_colormapped_dict():
    """Colors passed as dicts are treated as colormapped"""
    color = {0: np.array([1, 1, 1, 1]), 1: np.array([1, 1, 0, 1])}
    properties = {
        'hello': np.array([1, 1, 1, 1]),
        'hi': np.array([1, 0, 0, 1]),
    }
    assert is_color_mapped(color, properties)


def test_is_colormapped_array():
    """Colors passed as list/array are treated as not colormapped"""
    color_list = [[1, 1, 1, 1], [1, 1, 0, 1]]
    properties = {
        'hello': np.array([1, 1, 1, 1]),
        'hi': np.array([1, 0, 0, 1]),
    }
    assert not is_color_mapped(color_list, properties)

    color_array = np.array(color_list)
    assert not is_color_mapped(color_array, properties)
