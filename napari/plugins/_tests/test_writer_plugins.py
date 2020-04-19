import os

import numpy as np
import pytest

from napari.layers import Image, Points
from napari.plugins._builtins import write_layer_data_with_plugins
from napari.plugins.io import write_layers_with_plugins
from napari.plugins.exceptions import PluginCallError
from napari.utils import io


def test_builtin_write_image(tmpdir):
    """Test the builtin writer plugin writes an image layer."""
    layer = Image(np.random.rand(20, 20), name='image')

    path = os.path.join(tmpdir, 'image_file.tif')

    # Check file does not exist
    assert not os.path.isfile(path)

    # Write data
    write_layers_with_plugins(path, layer, plugin_name='builtins')

    # Check file now exists
    assert os.path.isfile(path)

    # Read image data
    data = io.imread(path)

    # Compare read data to data on layer
    np.testing.assert_allclose(layer.data, data)


def test_builtin_write_points(tmpdir):
    """Test the builtin writer plugin writes a points layer."""
    layer = Points(np.random.rand(20, 2), name='points')

    path = os.path.join(tmpdir, 'points_file.csv')

    # Check file does not exist
    assert not os.path.isfile(path)

    # Write data
    write_layers_with_plugins(path, layer, plugin_name='builtins')

    # Check file now exists
    assert os.path.isfile(path)

    # Read points data
    data, column_names = io.read_csv(path)

    # Compare read data to data on layer
    np.testing.assert_allclose(layer.data, data[:, 1:])
    # Check points index have been added appropriately
    np.testing.assert_allclose(list(range(len(layer.data))), data[:, 0])
    assert column_names == ['index', 'axis-0', 'axis-1']


def test_builtin_get_writer(tmpdir):
    """Test the builtin writer plugin writes layers to a folder."""

    img = Image(np.random.rand(20, 20), name='image')
    pts = Points(np.random.rand(20, 2), name='points')

    # Check folder exists but files do not
    assert os.path.isdir(tmpdir)
    assert not os.path.isfile(os.path.join(tmpdir, 'image.tif'))
    assert not os.path.isfile(os.path.join(tmpdir, 'points.csv'))

    # Write data
    write_layers_with_plugins(tmpdir, [img, pts], plugin_name='builtins')

    # Check folder and files exist
    assert os.path.isdir(tmpdir)
    assert os.path.isfile(os.path.join(tmpdir, 'image.tif'))
    assert os.path.isfile(os.path.join(tmpdir, 'points.csv'))


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
