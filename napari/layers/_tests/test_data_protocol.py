import pytest

from napari._tests.utils import layer_test_data
from napari.layers import Shapes, Surface
from napari.layers._data_protocols import assert_protocol

EASY_TYPES = [i for i in layer_test_data if i[0] not in (Shapes, Surface)]


def _layer_test_data_id(test_data):
    LayerCls, data, ndim = test_data
    objtype = type(data).__name__
    dtype = getattr(data, 'dtype', '?')
    return f'{LayerCls.__name__}_{objtype}_{dtype}_{ndim}d'


@pytest.mark.parametrize('test_data', EASY_TYPES, ids=_layer_test_data_id)
def test_layer_protocol(test_data):
    LayerCls, data, _ = test_data
    layer = LayerCls(data)
    assert_protocol(layer.data)


def test_layer_protocol_raises():
    with pytest.raises(TypeError) as e:
        assert_protocol([])  # list doesn't provide the protocol
    assert "Missing methods: " in str(e)
    assert "'shape'" in str(e)
