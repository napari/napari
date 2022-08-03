from pathlib import Path
from typing import Callable, Optional

import imageio
import npe2
import numpy as np
import pytest
import tifffile

from napari_builtins.io._write import write_csv


@pytest.fixture
def save_image(tmp_path: Path):
    """Create a temporary file."""

    def _save(filename: str, data: Optional[np.ndarray] = None):
        dest = tmp_path / filename
        _data: np.ndarray = np.random.rand(20, 20) if data is None else data
        if dest.suffix in {".tif", ".tiff"}:
            tifffile.imwrite(str(dest), _data)
        elif dest.suffix in {'.npy'}:
            np.save(str(dest), _data)
        else:
            imageio.imsave(str(dest), _data)
        return dest

    return _save


@pytest.mark.parametrize('ext', ['.tif', '.npy', '.png', '.jpg'])
@pytest.mark.parametrize('stack', [False, True])
def test_reader_plugin_tif(save_image: Callable[..., Path], ext, stack):
    """Test the builtin reader plugin reads a temporary file."""
    files = [
        str(save_image(f'test_{i}{ext}')) for i in range(5 if stack else 1)
    ]
    layer_data = npe2.read(files, stack=stack)
    assert isinstance(layer_data, list)
    assert len(layer_data) == 1
    assert isinstance(layer_data[0], tuple)


def test_reader_plugin_url():
    layer_data = npe2.read(
        ['https://samples.fiji.sc/FakeTracks.tif'], stack=False
    )
    assert isinstance(layer_data, list)
    assert len(layer_data) == 1
    assert isinstance(layer_data[0], tuple)


def test_reader_plugin_csv(tmp_path):
    """Test the builtin reader plugin reads a temporary file."""
    dest = str(tmp_path / 'test.csv')
    table = np.random.random((5, 3))
    write_csv(dest, table, column_names=['index', 'axis-0', 'axis-1'])

    layer_data = npe2.read([dest], stack=False)

    assert layer_data is not None
    assert isinstance(layer_data, list)
    assert len(layer_data) == 1
    assert isinstance(layer_data[0], tuple)
    assert layer_data[0][2] == 'points'
    assert np.allclose(table[:, 1:], layer_data[0][0])
