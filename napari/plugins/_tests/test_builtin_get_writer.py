import os

import pytest

from napari.plugins._builtins import napari_get_writer
from napari.plugins.exceptions import PluginCallError


def test_get_writer(plugin_manager, tmpdir, layer_data_and_types):
    """Test writing layers data."""
    _, layer_data, layer_types, filenames = layer_data_and_types
    path = os.path.join(tmpdir, 'layers_folder')

    writer = napari_get_writer(path, layer_types)
    # Write data
    assert writer(path, layer_data, plugin_manager)

    # Check folder and files exist
    assert os.path.isdir(path)
    for f in filenames:
        assert os.path.isfile(os.path.join(path, f))

    assert set(os.listdir(path)) == set(filenames)
    assert set(os.listdir(tmpdir)) == set(['layers_folder'])


def test_get_writer_bad_plugin(
    plugin_manager, temporary_hookimpl, tmpdir, layer_data_and_types
):
    """Test cleanup when get_writer has an exception."""

    def bad_write_points(path, data, meta):
        raise ValueError("shoot!")

    _, layer_data, layer_types, filenames = layer_data_and_types

    # this time we try writing directly to the tmpdir (which already exists)
    writer = napari_get_writer(tmpdir, layer_types)
    # call writer with a bad hook implementation inserted
    with temporary_hookimpl(bad_write_points, 'napari_write_points'):
        with pytest.raises(PluginCallError):
            writer(tmpdir, layer_data, plugin_manager)

    # should have deleted all new files, but not the tmpdir
    assert os.path.isdir(tmpdir)
    for f in filenames:
        assert not os.path.isfile(os.path.join(tmpdir, f))

    # now try writing to a nested folder inside of tmpdir
    path = os.path.join(tmpdir, 'layers_folder')
    writer = napari_get_writer(path, layer_types)
    # call writer with a bad hook implementation inserted
    with temporary_hookimpl(bad_write_points, 'napari_write_points'):
        with pytest.raises(PluginCallError):
            writer(path, layer_data, plugin_manager)

    # should have deleted the new nested folder, but not the tmpdir
    assert os.path.isdir(tmpdir)
    assert not os.path.exists(path)
