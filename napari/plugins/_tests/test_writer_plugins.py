import os
from tempfile import TemporaryDirectory
import numpy as np

from napari.utils import io
from napari.plugins.io import write_layers_with_plugins
from napari.components import ViewerModel as Viewer


def test_builtin_get_writer(viewer_factory):
    """Test the builtin writer plugin writes layers to a folder."""
    viewer = Viewer()
    viewer.add_image(np.random.rand(20, 20), name='image')
    viewer.add_points(np.random.rand(20, 2), name='points')

    with TemporaryDirectory() as fout:
        # Check folder exists but files do not
        assert os.path.isdir(fout)
        assert not os.path.isfile(os.path.join(fout, 'image.tif'))
        assert not os.path.isfile(os.path.join(fout, 'points.csv'))

        # Write data
        write_layers_with_plugins(fout, viewer.layers, plugin_name='builtins')

        # Check folder and files exist
        assert os.path.isdir(fout)
        assert os.path.isfile(os.path.join(fout, 'image.tif'))
        assert os.path.isfile(os.path.join(fout, 'points.csv'))


def test_builtin_write_image(viewer_factory):
    """Test the builtin writer plugin writes an image layer."""
    viewer = Viewer()
    layer = viewer.add_image(np.random.rand(20, 20), name='image')

    with TemporaryDirectory() as fout:
        path = os.path.join(fout, 'image_file.tif')

        # Check file does not exist
        assert not os.path.isfile(path)

        # Write data
        write_layers_with_plugins(path, layer, plugin_name='builtins')

        # Check file now exists
        assert os.path.isfile(path)

        # Read image data
        data = io.magicread(path)
        # Compare read data to data on layer
        np.testing.assert_allclose(layer.data, data)


def test_builtin_write_points(viewer_factory):
    """Test the builtin writer plugin writes a points layer."""
    viewer = Viewer()
    layer = viewer.add_points(np.random.rand(20, 2), name='points')

    with TemporaryDirectory() as fout:
        path = os.path.join(fout, 'points_file.csv')

        # Check file does not exist
        assert not os.path.isfile(path)

        # Write data
        write_layers_with_plugins(path, layer, plugin_name='builtins')

        # Check file now exists
        assert os.path.isfile(path)
