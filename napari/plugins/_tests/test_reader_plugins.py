from napari.plugins.utils import get_layer_data_from_plugins
from tempfile import NamedTemporaryFile
import numpy as np
from skimage import io
import os
from pathlib import Path


def test_iter_reader_plugins(plugin_manager):
    """Test safe iteration through reader plugins even with errors.

    `napari_bad_plugin2` is a plugin that loads fine but throws an error during
    file-reading.  this tests that we can gracefully handle that.
    """

    # the plugin loads fine, so there should be no exceptions yet.
    assert 'napari_bad_plugin2' not in plugin_manager._exceptions

    # we want 'napari_bad_plugin2' to be the first plugin called.  But
    # `napari_test_plugin` will be in line first... Until we can
    # reorder the call-order of plugins, (#1023), this line serves to prevent
    # that good plugin from running.
    plugin_manager.set_blocked('napari_test_plugin')

    # but when we try to read an image path, it will raise an IOError.
    # we want to catch and store that IOError, and then move on to give other
    # plugins chance to return layer_data
    layer_data = get_layer_data_from_plugins('image.ext', plugin_manager)

    # the good plugins (like "napari_test_plugin") should return layer_data
    assert layer_data

    # but the exception from `bad_plugin2` should now be stored.
    assert 'napari_bad_plugin2' in plugin_manager._exceptions
    # we can print out a string that should have the explanation of the error.
    exception_string = plugin_manager.format_exceptions('napari_bad_plugin2')
    assert 'IOError' in exception_string
    assert "napari_get_reader" in exception_string


def test_builtin_reader_plugin(viewer_factory, builtin_plugin_manager):
    """Test the builtin reader plugin reads a temporary file."""
    with NamedTemporaryFile(suffix='.tif', delete=False) as tmp:
        data = np.random.rand(20, 20)
        io.imsave(tmp.name, data)
        tmp.seek(0)

        layer_data = get_layer_data_from_plugins(
            tmp.name, builtin_plugin_manager
        )

        assert isinstance(layer_data, list)
        assert len(layer_data) == 1
        assert isinstance(layer_data[0], tuple)
        assert np.allclose(data, layer_data[0][0])

        view, viewer = viewer_factory()
        viewer.add_path(tmp.name)

        assert np.allclose(viewer.layers[0].data, data)


def test_builtin_reader_plugin_stacks(viewer_factory, builtin_plugin_manager):
    """Test the builtin reader plugin reads multiple files as a stack."""
    data = np.random.rand(5, 20, 20)
    tmps = []
    for plane in data:
        tmp = NamedTemporaryFile(suffix='.tif', delete=False)
        io.imsave(tmp.name, plane)
        tmp.seek(0)
        tmps.append(tmp)

    _, viewer = viewer_factory()
    # add_path should take both strings and Path object, so we make one of the
    # pathnames a Path object
    names = [tmp.name for tmp in tmps]
    names[0] = Path(names[0])
    viewer.add_path(names, stack=True)
    assert np.allclose(viewer.layers[0].data, data)
    for tmp in tmps:
        tmp.close()
        os.unlink(tmp.name)


def test_nonsense_path_is_ok(plugin_manager):
    """Test that a path with no readers doesn't throw an exception."""
    layer_data = get_layer_data_from_plugins('image.NONsense', plugin_manager)
    assert not layer_data
