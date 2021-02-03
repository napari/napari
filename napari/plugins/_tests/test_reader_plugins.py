import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from napari.components import ViewerModel
from napari.plugins.io import read_data_with_plugins
from napari.utils import io


def test_builtin_reader_plugin():
    """Test the builtin reader plugin reads a temporary file."""

    with NamedTemporaryFile(suffix='.tif', delete=False) as tmp:
        data = np.random.rand(20, 20)
        io.imsave(tmp.name, data)
        tmp.seek(0)
        layer_data = read_data_with_plugins(tmp.name)

        assert isinstance(layer_data, list)
        assert len(layer_data) == 1
        assert isinstance(layer_data[0], tuple)
        assert np.allclose(data, layer_data[0][0])

        viewer = ViewerModel()
        viewer.open(tmp.name, plugin='builtins')

        assert np.allclose(viewer.layers[0].data, data)


def test_builtin_reader_plugin_csv(tmpdir):
    """Test the builtin reader plugin reads a temporary file."""
    tmp = os.path.join(tmpdir, 'test.csv')
    column_names = ['index', 'axis-0', 'axis-1']
    table = np.random.random((5, 3))
    data = table[:, 1:]
    # Write csv file
    io.write_csv(tmp, table, column_names=column_names)
    layer_data = read_data_with_plugins(tmp)

    assert isinstance(layer_data, list)
    assert len(layer_data) == 1
    assert isinstance(layer_data[0], tuple)
    assert layer_data[0][2] == 'points'
    assert np.allclose(data, layer_data[0][0])

    viewer = ViewerModel()
    viewer.open(tmp, plugin='builtins')

    assert np.allclose(viewer.layers[0].data, data)


def test_builtin_reader_plugin_stacks():
    """Test the builtin reader plugin reads multiple files as a stack."""
    data = np.random.rand(5, 20, 20)
    tmps = []
    for plane in data:
        tmp = NamedTemporaryFile(suffix='.tif', delete=False)
        io.imsave(tmp.name, plane)
        tmp.seek(0)
        tmps.append(tmp)

    viewer = ViewerModel()
    # open should take both strings and Path object, so we make one of the
    # pathnames a Path object
    names = [tmp.name for tmp in tmps]
    names[0] = Path(names[0])
    viewer.open(names, stack=True, plugin='builtins')
    assert np.allclose(viewer.layers[0].data, data)
    for tmp in tmps:
        tmp.close()
        os.unlink(tmp.name)
