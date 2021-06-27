import os

import pytest
from napari_plugin_engine import PluginCallError

from napari.plugins import _builtins


# test_plugin_manager fixture is provided by napari_plugin_engine._testsupport
def test_get_writer_succeeds(
    napari_plugin_manager, tmpdir, layer_data_and_types
):
    """Test writing layers data."""

    _, layer_data, layer_types, filenames = layer_data_and_types
    path = os.path.join(tmpdir, 'layers_folder')

    writer = napari_plugin_manager.hook.napari_get_writer(
        path=path, layer_types=layer_types
    )

    # Write data
    assert writer == _builtins.write_layer_data_with_plugins
    assert writer(path, layer_data, plugin_name=None)

    # Check folder and files exist
    assert os.path.isdir(path)
    for f in filenames:
        assert os.path.isfile(os.path.join(path, f))

    assert set(os.listdir(path)) == set(filenames)
    assert set(os.listdir(tmpdir)) == {'layers_folder'}


# the layer_data_and_types fixture is defined in napari/conftest.py
# test_plugin_manager fixture is provided by napari_plugin_engine._testsupport
def test_get_writer_bad_plugin(
    napari_plugin_manager, tmpdir, layer_data_and_types
):
    """Test cleanup when get_writer has an exception."""
    from napari_plugin_engine import napari_hook_implementation

    class bad_plugin:
        @napari_hook_implementation
        def napari_write_points(path, data, meta):
            raise ValueError("shoot!")

    _, layer_data, layer_types, filenames = layer_data_and_types

    napari_plugin_manager.register(bad_plugin)
    # this time we try writing directly to the tmpdir (which already exists)
    writer = _builtins.napari_get_writer(tmpdir, layer_types)

    # call writer with a bad hook implementation inserted
    with pytest.raises(PluginCallError):
        writer(tmpdir, layer_data, plugin_name=None)

    # should have deleted all new files, but not the tmpdir
    assert os.path.isdir(tmpdir)
    for f in filenames:
        assert not os.path.isfile(os.path.join(tmpdir, f))

    # now try writing to a nested folder inside of tmpdir
    path = os.path.join(tmpdir, 'layers_folder')
    writer = _builtins.napari_get_writer(path, layer_types)

    # call writer with a bad hook implementation inserted
    with pytest.raises(PluginCallError):
        writer(tmpdir, layer_data, plugin_name=None)

    # should have deleted the new nested folder, but not the tmpdir
    assert os.path.isdir(tmpdir)
    assert not os.path.exists(path)
