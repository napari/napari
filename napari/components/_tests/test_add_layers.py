from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from napari_plugin_engine import HookImplementation

from napari._tests.utils import layer_test_data
from napari.components.viewer_model import ViewerModel
from napari.layers._source import Source

img = np.random.rand(10, 10)
layer_data = [(lay[1], {}, lay[0].__name__.lower()) for lay in layer_test_data]


def _impl(path):
    """just a dummy Hookimpl object to return from mocks"""
    pass


_testimpl = HookImplementation(_impl, plugin_name='testimpl')


@pytest.mark.parametrize("layer_datum", layer_data)
def test_add_layers_with_plugins(layer_datum):
    """Test that add_layers_with_plugins adds the expected layer types."""
    with patch(
        "napari.plugins.io.read_data_with_plugins",
        MagicMock(return_value=([layer_datum], _testimpl)),
    ):
        v = ViewerModel()
        v._add_layers_with_plugins(['mock_path'], stack=False)
        layertypes = [layer._type_string for layer in v.layers]
        assert layertypes == [layer_datum[2]]

        expected_source = Source(path='mock_path', reader_plugin='testimpl')
        assert all(lay.source == expected_source for lay in v.layers)


@patch(
    "napari.plugins.io.read_data_with_plugins",
    MagicMock(return_value=([], _testimpl)),
)
def test_plugin_returns_nothing():
    """Test that a plugin returning nothing adds nothing to the Viewer."""
    v = ViewerModel()
    v._add_layers_with_plugins(['mock_path'], stack=False)
    assert not v.layers


@patch(
    "napari.plugins.io.read_data_with_plugins",
    MagicMock(return_value=([(img,)], _testimpl)),
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

    expected_source = Source(path='mock_path.tif', reader_plugin='testimpl')
    assert all(lay.source == expected_source for lay in viewer.layers)


def test_viewer_open_no_plugin(tmp_path):
    viewer = ViewerModel()
    fname = tmp_path / 'gibberish.gbrsh'
    fname.touch()
    with pytest.raises(ValueError, match='No plugin found capable of reading'):
        # will default to builtins
        viewer.open(fname)


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
        "napari.plugins.io.read_data_with_plugins",
        MagicMock(return_value=(layer_data, _testimpl)),
    ):

        v = ViewerModel()
        v._add_layers_with_plugins(['mock_path'], kwargs=kwargs, stack=False)
        expected_source = Source(path='mock_path', reader_plugin='testimpl')
        for layer in v.layers:
            for key, val in kwargs.items():
                assert getattr(layer, key) == val
                # if plugins don't provide "name", it falls back to path name
                if 'name' not in kwargs:
                    assert layer.name.startswith('mock_path')
            assert layer.source == expected_source


def test_add_points_layer_with_different_range_updates_all_slices():
    """See https://github.com/napari/napari/pull/4819"""
    viewer = ViewerModel()
    point = viewer.add_points([[10, 5, 5]])
    np.testing.assert_array_equal(point._indices_view, [0])

    other_point = viewer.add_points([[8, 1, 1]])
    np.testing.assert_array_equal(point._indices_view, [])
    np.testing.assert_array_equal(other_point._indices_view, [0])

    viewer.dims.set_point(range(len(viewer.layers._ranges)), (10, 5, 5))
    np.testing.assert_array_equal(point._indices_view, [0])
    np.testing.assert_array_equal(other_point._indices_view, [])

    viewer.layers.remove(point)
    np.testing.assert_array_equal(other_point._indices_view, [0])
