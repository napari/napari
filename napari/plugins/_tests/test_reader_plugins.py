from napari.plugins.utils import (
    get_layer_data_from_plugins,
    permute_hook_implementations,
)
from tempfile import NamedTemporaryFile
import numpy as np
from skimage import io
from napari import Viewer


def test_iter_reader_plugins(plugin_manager):
    """Test safe iteration through reader plugins even with errors."""

    # first we move one of the "bad" plugins to the front of the line
    # napari_bad_plugin2 returns a successful "napari_get_reader"
    # but then raises an IOError
    permute_hook_implementations(
        plugin_manager.hook.napari_get_reader, ['napari_bad_plugin2']
    )

    # the plugin loads fine, so there should be no exceptions yet.
    assert 'napari_bad_plugin2' not in plugin_manager._exceptions

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


def test_builtin_reader_plugin(qtbot, builtin_plugin_manager):
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

        viewer = Viewer()
        qtbot.addWidget(viewer.window.qt_viewer)
        viewer.add_path(tmp.name)

        assert np.allclose(viewer.layers[0].data, data)
        viewer.window.close()


def test_nonsense_path_is_ok(plugin_manager):
    """Test that a path with no readers doesn't throw an exception."""
    layer_data = get_layer_data_from_plugins('image.NONsense', plugin_manager)
    assert not layer_data
