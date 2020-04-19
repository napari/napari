import os
import numpy as np

from napari.utils import io
from napari.plugins.io import write_layers_with_plugins
from napari.components import ViewerModel as Viewer


def test_builtin_write_image(viewer_factory, tmpdir):
    """Test the builtin writer plugin writes an image layer."""
    viewer = Viewer()
    layer = viewer.add_image(np.random.rand(20, 20), name='image')

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


def test_builtin_write_points(viewer_factory, tmpdir):
    """Test the builtin writer plugin writes a points layer."""
    viewer = Viewer()
    layer = viewer.add_points(np.random.rand(20, 2), name='points')

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


def test_builtin_get_writer(viewer_factory, tmpdir):
    """Test the builtin writer plugin writes layers to a folder."""
    viewer = Viewer()
    viewer.add_image(np.random.rand(20, 20), name='image')
    viewer.add_points(np.random.rand(20, 2), name='points')

    # Check folder exists but files do not
    assert os.path.isdir(tmpdir)
    assert not os.path.isfile(os.path.join(tmpdir, 'image.tif'))
    assert not os.path.isfile(os.path.join(tmpdir, 'points.csv'))

    # Write data
    write_layers_with_plugins(tmpdir, viewer.layers, plugin_name='builtins')

    # Check folder and files exist
    assert os.path.isdir(tmpdir)
    assert os.path.isfile(os.path.join(tmpdir, 'image.tif'))
    assert os.path.isfile(os.path.join(tmpdir, 'points.csv'))
