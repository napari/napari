import numpy as np
import pytest

from napari.layers.utils.color_manager import ColorManager


def test_color_manager_empty():
    cm = ColorManager()
    np.testing.assert_allclose(cm.colors, np.empty((0, 4)))


color_str = 'red'
color_list = [[1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]]
color_arr = np.asarray(color_list)


@pytest.mark.parametrize('color', [color_str, color_list, color_arr])
def test_set_color_direct(color):
    cm = ColorManager()
    cm.set_color(color=color, n_colors=3)

    expected_colors = np.array([[1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]])
    np.testing.assert_allclose(cm.colors, expected_colors)
