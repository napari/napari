import importlib
import os

import numpy as np
import pytest

from napari.layers import Image, Points
from napari.plugins._builtins import (
    napari_get_writer,
    write_layer_data_with_plugins,
)
from napari.plugins.exceptions import PluginCallError


def test_get_writer(tmpdir, layer_data_and_types):
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


def test_get_writer_bad_plugin(tmpdir, layer_data_and_types):
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


def test_write_layer_data_with_plugins_succeeds(
    plugin_manager, temporary_hookimpl, tmpdir
):
    img = Image(np.random.rand(20, 20), name='image')
    pts = Points(np.random.rand(20, 2), name='points')
    layer_data = [img.as_layer_data_tuple(), pts.as_layer_data_tuple()]

    write_layer_data_with_plugins(tmpdir, layer_data, plugin_manager)

    # Check folder and files exist
    assert os.path.isdir(tmpdir)
    assert os.path.isfile(os.path.join(tmpdir, 'image.tif'))
    assert os.path.isfile(os.path.join(tmpdir, 'points.csv'))
    # make sure the temporary directory inside write_layer_data_with_plugins
    # was cleaned up
    assert set(os.listdir(tmpdir)) == {'points.csv', 'image.tif'}


def test_write_layer_data_with_plugins_fails(
    plugin_manager, temporary_hookimpl, tmpdir
):
    def bad_write_points(path, data, meta):
        raise ValueError("shoot!")

    img = Image(np.random.rand(20, 20), name='image')
    pts = Points(np.random.rand(20, 2), name='points')
    layer_data = [img.as_layer_data_tuple(), pts.as_layer_data_tuple()]

    with temporary_hookimpl(bad_write_points, 'napari_write_points'):
        with pytest.raises(PluginCallError):
            write_layer_data_with_plugins(tmpdir, layer_data, plugin_manager)

    assert os.path.isdir(tmpdir)
    assert not os.path.isfile(os.path.join(tmpdir, 'points.csv'))
    assert not os.path.isfile(os.path.join(tmpdir, 'image.tif'))

    # if we create a new folder, make sure it also gets cleaned up:
    nested = os.path.join(tmpdir, 'inside')
    with temporary_hookimpl(bad_write_points, 'napari_write_points'):
        with pytest.raises(PluginCallError):
            write_layer_data_with_plugins(nested, layer_data, plugin_manager)

    assert not os.path.isdir(nested)
    assert not os.path.isfile(os.path.join(nested, 'points.csv'))
    assert not os.path.isfile(os.path.join(nested, 'image.tif'))
