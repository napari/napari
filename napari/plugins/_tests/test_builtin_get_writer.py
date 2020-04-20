import importlib
import os
import numpy as np
import pytest
from napari.plugins.exceptions import PluginCallError
from napari.plugins._builtins import napari_get_writer
from napari.layers import Image, Points


@pytest.fixture
def layer_data_and_types():
    layers = [
        Image(np.random.rand(20, 20)),
        Image(np.random.rand(20, 20)),
        Points(np.random.rand(20, 2)),
        Points(
            np.random.rand(20, 2), properties={'values': np.random.rand(20)}
        ),
    ]
    extensions = ['.tif', '.tif', '.csv', '.csv']
    layer_data = [l.as_layer_data_tuple() for l in layers]
    layer_types = [ld[2] for ld in layer_data]
    filenames = [l.name + e for l, e in zip(layers, extensions)]
    return layer_data, layer_types, filenames


def test_get_writer(tmpdir, layer_data_and_types):
    """Test writing layers data."""
    # make individual write layer builtin plugins get called first
    from napari.plugins import plugin_manager

    plugin_manager.hooks.napari_write_image.bring_to_front(['builtins'])
    plugin_manager.hooks.napari_write_points.bring_to_front(['builtins'])

    layer_data, layer_types, filenames = layer_data_and_types

    path = os.path.join(tmpdir, 'layers_folder')

    writer = napari_get_writer(path, layer_types)

    assert writer is not None

    # Check folder does not exist
    assert not os.path.isdir(path)

    # Write data
    assert writer(path, layer_data)

    # Check folder now exists
    assert os.path.isdir(path)

    # Check individual files now exist
    for f in filenames:
        assert os.path.isfile(os.path.join(path, f))

    # Check no additional files exist
    assert set(os.listdir(path)) == set(filenames)
    assert set(os.listdir(tmpdir)) == set(['layers_folder'])


def test_get_writer_bad_plugin(tmpdir, layer_data_and_types):
    """Test writing layers data."""
    # make individual write layer builtin plugins get called first
    from napari.plugins import plugin_manager

    plugin_manager.hooks.napari_write_image.bring_to_front(['builtins'])
    bad_plugin_path = 'napari.plugins._tests.fixtures.napari_bad_plugin'
    bad = importlib.import_module(bad_plugin_path)
    plugin_manager.register(bad)
    plugin_manager.hooks.napari_write_points.bring_to_front([bad_plugin_path])

    layer_data, layer_types, filenames = layer_data_and_types

    path = os.path.join(tmpdir, 'layers_folder')

    writer = napari_get_writer(path, layer_types)

    assert writer is not None

    # Check folder does not exist
    assert not os.path.isdir(path)

    # Write data
    with pytest.raises(PluginCallError):
        writer(path, layer_data)

    # Check folder still does not exist
    assert not os.path.isdir(path)

    # Check individual files still do not exist
    for f in filenames:
        assert not os.path.isfile(os.path.join(path, f))

    # Check no additional files exist
    assert set(os.listdir(tmpdir)) == set('')
