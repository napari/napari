import numpy as np

from napari.utils.colormaps.categorical_colormap_utils import (
    ColorCycle,
    compare_colormap_dicts,
)


def test_color_cycle():
    color_values = np.array([[1, 0, 0, 1], [0, 0, 1, 1]])
    cc_1 = ColorCycle(values=color_values)
    cc_2 = ColorCycle(values=color_values)

    np.testing.assert_allclose(cc_1.values, color_values)
    assert cc_1 == cc_2

    other_color_values = np.array([[1, 0, 0, 1], [1, 1, 1, 1]])
    cc_3 = ColorCycle(values=other_color_values)
    assert cc_1 != cc_3

    # verify that checking equality against another type works
    assert cc_1 != color_values


def test_compare_colormap_dicts():
    cmap_dict_1 = {
        0: np.array([0, 0, 0, 1]),
        1: np.array([1, 1, 1, 1]),
        2: np.array([1, 0, 0, 1]),
    }
    cmap_dict_2 = {
        0: np.array([0, 0, 0, 1]),
        1: np.array([1, 1, 1, 1]),
        2: np.array([1, 0, 0, 1]),
    }
    assert compare_colormap_dicts(cmap_dict_1, cmap_dict_2)

    # same keys different values
    cmap_dict_3 = {
        0: np.array([1, 1, 1, 1]),
        1: np.array([1, 1, 1, 1]),
        2: np.array([1, 0, 0, 1]),
    }
    assert not compare_colormap_dicts(cmap_dict_1, cmap_dict_3)

    # different number of keys
    cmap_dict_4 = {
        0: np.array([1, 1, 1, 1]),
        1: np.array([1, 1, 1, 1]),
    }
    assert not compare_colormap_dicts(cmap_dict_1, cmap_dict_4)
    assert not compare_colormap_dicts(cmap_dict_3, cmap_dict_4)

    # same number of keys, but different keys
    cmap_dict_5 = {
        'hi': np.array([1, 1, 1, 1]),
        'hello': np.array([1, 1, 1, 1]),
        'hallo': np.array([1, 0, 0, 1]),
    }
    assert not compare_colormap_dicts(cmap_dict_1, cmap_dict_5)
    assert not compare_colormap_dicts(cmap_dict_3, cmap_dict_5)
