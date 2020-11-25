from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from napari._tests.utils import layer_test_data
from napari.components.viewer_model import ViewerModel

img = np.random.rand(10, 10)
layer_data = [(lay[1], {}, lay[0].__name__.lower()) for lay in layer_test_data]


@pytest.mark.parametrize("layer_datum", layer_data)
def test_add_layers_with_plugins(layer_datum):
    """Test that add_layers_with_plugins adds the expected layer types."""
    with patch(
        "napari.components.viewer_model.read_data_with_plugins",
        MagicMock(return_value=[layer_datum]),
    ):
        v = ViewerModel()
        v._add_layers_with_plugins('mock_path')
        layertypes = [layer._type_string for layer in v.layers]
        assert layertypes == [layer_datum[2]]


@patch(
    "napari.components.viewer_model.read_data_with_plugins",
    MagicMock(return_value=[]),
)
def test_plugin_returns_nothing():
    """Test that a plugin returning nothing adds nothing to the Viewer."""
    v = ViewerModel()
    v._add_layers_with_plugins('mock_path')
    assert not v.layers


@patch(
    "napari.components.viewer_model.read_data_with_plugins",
    MagicMock(return_value=[(img,)]),
)
def test_viewer_open():
    """Test that a plugin to returning an image adds stuff to the viewer."""
    viewer = ViewerModel()
    assert len(viewer.layers) == 0
    viewer.open('mock_path.tif')
    assert len(viewer.layers) == 1
    # The name should be taken from the path name, stripped of extension
    assert viewer.layers[0].name == 'mock_path'

    # stack=True also works... and very long names are truncated
    viewer.open('mock_path.tif', stack=True)
    assert len(viewer.layers) == 2
    assert viewer.layers[1].name.startswith('mock_path')


plugin_returns = [
    ([(img, {'name': 'foo'})], {'name': 'bar'}),
    ([(img, {'blending': 'additive'}), (img,)], {'blending': 'translucent'}),
]


@pytest.mark.parametrize("layer_data, kwargs", plugin_returns)
def test_add_layers_with_plugins_and_kwargs(layer_data, kwargs):
    """Test that _add_layers_with_plugins kwargs override plugin kwargs.

    see also: napari.components._test.test_prune_kwargs
    """
    with patch(
        "napari.components.viewer_model.read_data_with_plugins",
        MagicMock(return_value=layer_data),
    ):
        v = ViewerModel()
        v._add_layers_with_plugins('mock_path', kwargs=kwargs)
        for layer in v.layers:
            for key, val in kwargs.items():
                assert getattr(layer, key) == val
                # if plugins don't provide "name", it falls back to path name
                if 'name' not in kwargs:
                    assert layer.name.startswith('mock_path')
