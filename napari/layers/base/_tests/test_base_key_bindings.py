import pytest

from napari.layers.base import _base_key_bindings as key_bindings
from napari.layers.points import Points


def test_hold_to_pan_zoom(layer):
    data = [[1, 3], [8, 4], [10, 10], [15, 4]]
    layer = Points(data, size=1)

    layer.mode = 'transform'
    # need to go through the generator
    gen = key_bindings.hold_to_pan_zoom(layer)
    assert layer.mode == 'transform'
    next(gen)
    assert layer.mode == 'pan_zoom'
    with pytest.raises(StopIteration):
        next(gen)
    assert layer.mode == 'transform'
