import importlib
import os
import pytest
from napari.plugins.exceptions import PluginCallError
from napari.plugins._builtins import napari_get_writer
from napari.plugins._tests.fixtures.layer_data import (  # noqa: F401
    layer_data_and_types,
)


def test_get_writer(tmpdir, layer_data_and_types):  # noqa: F811
    """Test writing layers data."""
    # make individual write layer builtin plugins get called first
    from napari.plugins import plugin_manager

    plugin_manager.hooks.napari_write_image.bring_to_front(['builtins'])
    plugin_manager.hooks.napari_write_points.bring_to_front(['builtins'])

    _, layer_data, layer_types, filenames = layer_data_and_types

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


def test_get_writer_bad_plugin(tmpdir, layer_data_and_types):  # noqa: F811
    """Test writing layers data."""
    # make individual write layer builtin plugins get called first
    from napari.plugins import plugin_manager

    plugin_manager.hooks.napari_write_image.bring_to_front(['builtins'])
    bad_plugin_path = 'napari.plugins._tests.fixtures.napari_bad_plugin'
    bad = importlib.import_module(bad_plugin_path)
    plugin_manager.register(bad)
    plugin_manager.hooks.napari_write_points.bring_to_front([bad_plugin_path])

    _, layer_data, layer_types, filenames = layer_data_and_types

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
