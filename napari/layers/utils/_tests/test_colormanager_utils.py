import numpy as np
import pytest

from napari.layers.utils._color_manager_utils import is_color_mapped

properties = {
    'hello': [1, 2, 3, 4],
    'hai': [1, 2, 3, 4],
    'hey': [1, 2, 3, 4],
}


def test_is_color_mapped_str():
    result = is_color_mapped('hello', properties)
    assert result is True

    result = is_color_mapped('red', properties)
    assert result is False


def test_is_color_mapped_dict():
    color = {1: 'red', 2: 'blue'}
    result = is_color_mapped(color, properties)
    assert result is True


color_array_list = [1, 1, 1, 1]
color_array_numpy = np.asarray(color_array_list)


@pytest.mark.parametrize("color_array", [color_array_list, color_array_numpy])
def test_is_color_mapped_array(color_array):
    result = is_color_mapped(color_array, properties)
    assert result is False


def test_invalid_is_color_mapped():
    with pytest.raises(TypeError):
        is_color_mapped(42, properties)
