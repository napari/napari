import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
from napari.utils import io

from napari.plugins.exceptions import PLUGIN_ERRORS, format_exceptions
from napari.plugins.io import read_data_with_plugins


def test_iter_reader_plugins(plugin_manager):
    """Test safe iteration through reader plugins even with errors.

    `napari_bad_plugin2` is a plugin that loads fine but throws an error during
    file-reading.  this tests that we can gracefully handle that.
    """

    # the plugin loads fine, so there should be no exceptions yet.
    assert 'napari_bad_plugin2' not in PLUGIN_ERRORS

    # make sure 'napari_bad_plugin2' gets called first
    plugin_manager.hooks.napari_get_reader.bring_to_front(
        ['napari_bad_plugin2']
    )

    # but when we try to read an image path, it will raise an IOError.
    # we want to catch and store that IOError, and then move on to give other
    # plugins chance to return layer_data
    layer_data = read_data_with_plugins(
        'image.ext', plugin_manager=plugin_manager
    )

    # the good plugins (like "napari_test_plugin") should return layer_data
    assert layer_data

    # but the exception from `bad_plugin2` should now be stored.
    assert 'napari_bad_plugin2' in PLUGIN_ERRORS
    # we can print out a string that should have the explanation of the error.
    exception_string = format_exceptions('napari_bad_plugin2')
    assert 'IOError' in exception_string
    assert "napari_get_reader" in exception_string


def test_builtin_reader_plugin(viewer_factory):
    """Test the builtin reader plugin reads a temporary file."""
    from napari.plugins import plugin_manager

    plugin_manager.hooks.napari_get_reader.bring_to_front(['builtins'])

    with NamedTemporaryFile(suffix='.tif', delete=False) as tmp:
        data = np.random.rand(20, 20)
        io.imsave(tmp.name, data)
        tmp.seek(0)
        layer_data = read_data_with_plugins(tmp.name)

        assert isinstance(layer_data, list)
        assert len(layer_data) == 1
        assert isinstance(layer_data[0], tuple)
        assert np.allclose(data, layer_data[0][0])

        view, viewer = viewer_factory()
        viewer.open(tmp.name)

        assert np.allclose(viewer.layers[0].data, data)


def test_builtin_reader_plugin_stacks(viewer_factory):
    """Test the builtin reader plugin reads multiple files as a stack."""
    from napari.plugins import plugin_manager

    plugin_manager.hooks.napari_get_reader.bring_to_front(['builtins'])

    data = np.random.rand(5, 20, 20)
    tmps = []
    for plane in data:
        tmp = NamedTemporaryFile(suffix='.tif', delete=False)
        io.imsave(tmp.name, plane)
        tmp.seek(0)
        tmps.append(tmp)

    _, viewer = viewer_factory()
    # open should take both strings and Path object, so we make one of the
    # pathnames a Path object
    names = [tmp.name for tmp in tmps]
    names[0] = Path(names[0])
    viewer.open(names, stack=True)
    assert np.allclose(viewer.layers[0].data, data)
    for tmp in tmps:
        tmp.close()
        os.unlink(tmp.name)


def test_nonsense_path_is_ok(plugin_manager):
    """Test that a path with no readers doesn't throw an exception."""
    layer_data = read_data_with_plugins(
        'image.NONsense', plugin_manager=plugin_manager
    )
    assert not layer_data
