from typing import TYPE_CHECKING

import numpy as np
import pytest

from napari._tests.utils import layer_test_data
from napari.components.viewer_model import ViewerModel
from napari.layers._source import Source

if TYPE_CHECKING:
    from npe2 import DynamicPlugin

layer_data = [(lay[1], {}, lay[0].__name__.lower()) for lay in layer_test_data]


@pytest.mark.parametrize("layer_datum", layer_data)
def test_add_layers_with_plugins(
    tmp_path, layer_datum, tmp_plugin: DynamicPlugin
):
    """Test that add_layers_with_plugins adds the expected layer types."""

    @tmp_plugin.contribute.reader(filename_patterns=['*.gbrsh'])
    def read(path):
        return lambda p: [layer_datum]

    mock_path = tmp_path / 'mock_path.gbrsh'
    mock_path.touch()

    v = ViewerModel()
    v._add_layers_with_plugins([str(mock_path)], stack=False)
    layertypes = [layer._type_string for layer in v.layers]
    assert layertypes == [layer_datum[2]]

    expected_source = Source(
        path=str(mock_path), reader_plugin=tmp_plugin.name
    )
    assert all(lay.source == expected_source for lay in v.layers)


def test_plugin_returns_nothing(tmp_path, tmp_plugin):
    """Test that a plugin returning nothing adds nothing to the Viewer."""

    @tmp_plugin.contribute.reader(filename_patterns=['*.gbrsh'])
    def read(path):
        return lambda p: [(None,)]

    mock_path = tmp_path / 'mock_path.gbrsh'
    mock_path.touch()
    v = ViewerModel()
    v._add_layers_with_plugins([str(mock_path)], stack=False)
    assert not v.layers


def test_viewer_open(tmp_plugin, tmp_path):
    """Test that a plugin to returning an image adds stuff to the viewer."""

    @tmp_plugin.contribute.reader(filename_patterns=['*.gbrsh'])
    def read(path):
        return lambda p: [(np.random.rand(10, 10),)]

    mock_path = tmp_path / 'mock_path.gbrsh'
    mock_path.touch()

    viewer = ViewerModel()
    assert len(viewer.layers) == 0
    viewer.open(mock_path, plugin=tmp_plugin.name)
    assert len(viewer.layers) == 1
    # The name should be taken from the path name, stripped of extension
    assert viewer.layers[0].name == 'mock_path'

    # stack=True also works... and very long names are truncated
    viewer.open(mock_path, stack=True, plugin=tmp_plugin.name)
    assert len(viewer.layers) == 2
    assert viewer.layers[1].name.startswith('mock_path')

    expected_source = Source(
        path=str(mock_path), reader_plugin=tmp_plugin.name
    )
    assert all(lay.source == expected_source for lay in viewer.layers)


def test_viewer_open_no_plugin(tmp_path):
    viewer = ViewerModel()
    fname = tmp_path / 'gibberish.gbrsh'
    fname.touch()
    with pytest.raises(
        ValueError, match="Plugin 'napari' not capable of reading"
    ):
        # will default to builtins
        viewer.open(fname)


img = np.random.rand(10, 10)
plugin_returns = [
    ([(img, {'name': 'foo'})], {'name': 'bar'}),
    ([(img, {'blending': 'additive'}), (img,)], {'blending': 'translucent'}),
]


@pytest.mark.parametrize("layer_data, kwargs", plugin_returns)
def test_add_layers_with_plugins_and_kwargs(
    tmp_path, tmp_plugin, layer_data, kwargs
):
    """Test that _add_layers_with_plugins kwargs override plugin kwargs.

    see also: napari.components._test.test_prune_kwargs
    """

    @tmp_plugin.contribute.reader(filename_patterns=['*.gbrsh'])
    def read(path):
        return lambda p: layer_data

    v = ViewerModel()
    fname = tmp_path / 'gibberish.gbrsh'
    fname.touch()
    v._add_layers_with_plugins([str(fname)], kwargs=kwargs, stack=False)
    expected_source = Source(path=str(fname), reader_plugin=tmp_plugin.name)
    for layer in v.layers:
        for key, val in kwargs.items():
            assert getattr(layer, key) == val
            # if plugins don't provide "name", it falls back to path name
            if 'name' not in kwargs:
                assert layer.name.startswith(str(fname.stem))
        assert layer.source == expected_source
