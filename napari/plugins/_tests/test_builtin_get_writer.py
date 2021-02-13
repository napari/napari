import os

import pytest
from napari_plugin_engine import PluginCallError

from napari.plugins import hook_specifications
from napari.plugins._builtins import (
    napari_get_writer,
    napari_write_image,
    napari_write_points,
    write_layer_data_with_plugins,
)


# test_plugin_manager fixture is provided by napari_plugin_engine._testsupport
def test_get_writer_succeeds(
    test_plugin_manager, tmpdir, layer_data_and_types, add_implementation
):
    """Test writing layers data."""
    test_plugin_manager.project_name = 'napari'
    test_plugin_manager.add_hookspecs(hook_specifications)

    _, layer_data, layer_types, filenames = layer_data_and_types
    path = os.path.join(tmpdir, 'layers_folder')

    add_implementation(napari_write_image)
    add_implementation(napari_write_points)
    add_implementation(napari_get_writer)
    writer = test_plugin_manager.hook.napari_get_writer(
        path=path, layer_types=layer_types
    )

    # Write data
    assert writer == write_layer_data_with_plugins
    assert writer(
        path, layer_data, plugin_name=None, plugin_manager=test_plugin_manager
    )

    # Check folder and files exist
    assert os.path.isdir(path)
    for f in filenames:
        assert os.path.isfile(os.path.join(path, f))

    assert set(os.listdir(path)) == set(filenames)
    assert set(os.listdir(tmpdir)) == {'layers_folder'}


# the layer_data_and_types fixture is defined in napari/conftest.py
# test_plugin_manager fixture is provided by napari_plugin_engine._testsupport
def test_get_writer_bad_plugin(
    test_plugin_manager, temporary_hookimpl, tmpdir, layer_data_and_types
):
    """Test cleanup when get_writer has an exception."""

    test_plugin_manager.project_name = 'napari'
    test_plugin_manager.add_hookspecs(hook_specifications)

    def bad_write_points(path, data, meta):
        raise ValueError("shoot!")

    _, layer_data, layer_types, filenames = layer_data_and_types

    # this time we try writing directly to the tmpdir (which already exists)
    writer = napari_get_writer(tmpdir, layer_types)
    # call writer with a bad hook implementation inserted
    with temporary_hookimpl(bad_write_points, 'napari_write_points'):
        with pytest.raises(PluginCallError):
            writer(
                tmpdir,
                layer_data,
                plugin_name=None,
                plugin_manager=test_plugin_manager,
            )

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
            writer(
                tmpdir,
                layer_data,
                plugin_name=None,
                plugin_manager=test_plugin_manager,
            )

    # should have deleted the new nested folder, but not the tmpdir
    assert os.path.isdir(tmpdir)
    assert not os.path.exists(path)
