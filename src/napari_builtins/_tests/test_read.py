from typing import TYPE_CHECKING

import imageio.v3 as iio
import npe2
import numpy as np
import pytest
import tifffile

from napari_builtins.io._read import (
    _read_python_source,
    _read_wavefront_obj_lines,
)
from napari_builtins.io._write import write_csv

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


@pytest.fixture
def save_image(tmp_path: 'Path'):
    """Create a temporary file."""

    def _save(filename: str, data: np.ndarray | None = None):
        dest = tmp_path / filename
        data_: np.ndarray = np.random.rand(20, 20) if data is None else data
        if filename.endswith(('png', 'jpg')):
            data_ = (data_ * 255).astype(np.uint8)
        if dest.suffix in {'.tif', '.tiff'}:
            tifffile.imwrite(str(dest), data_)
        elif dest.suffix in {'.npy'}:
            np.save(str(dest), data_)
        else:
            iio.imwrite(str(dest), data_)
        return dest

    return _save


@pytest.mark.parametrize('ext', ['.tif', '.npy', '.png', '.jpg'])
@pytest.mark.parametrize('stack', [False, True])
def test_reader_plugin_tif(save_image: 'Callable[..., Path]', ext, stack):
    """Test the builtin reader plugin reads a temporary file."""
    files = [
        str(save_image(f'test_{i}{ext}')) for i in range(5 if stack else 1)
    ]
    layer_data = npe2.read(files, stack=stack)
    assert isinstance(layer_data, list)
    assert len(layer_data) == 1
    assert isinstance(layer_data[0], tuple)


def test_animated_gif_reader(save_image):
    threeD_data = (np.random.rand(5, 20, 20, 3) * 255).astype(np.uint8)
    dest = save_image('animated.gif', threeD_data)
    layer_data = npe2.read([str(dest)], stack=False)
    assert len(layer_data) == 1
    assert layer_data[0][0].shape == (5, 20, 20, 3)


@pytest.mark.slow
def test_reader_plugin_url():
    layer_data = npe2.read(
        [
            'https://github.com/napari/docs/raw/main/docs/_static/images/nf-logo.png'
        ],
        stack=False,
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


@pytest.mark.parametrize(
    ('encoding', 'prefix', 'unit_value'),
    [
        ('utf-8', '', 'μm'),
        ('latin-1', '# coding: latin-1\n', 'µm'),
    ],
)
def test_read_python_source_uses_python_source_encoding(
    tmp_path, encoding, prefix, unit_value
):
    script_path = tmp_path / 'unit_script.py'
    script = f'{prefix}unit = {unit_value!r}\n'
    script_path.write_bytes(script.encode(encoding))

    assert _read_python_source(script_path) == script


def test_read_obj(tmp_path):
    obj_path = tmp_path / 'test.obj'
    with open(obj_path, 'w') as f:
        f.write("""
        # this should be ignored
        v 0 0 1
        v 1 0 2
        v 0 2 3
        f 0/9/6 1/4/5 2/7/8
        f 0 2 3
        f 0 1 2 3
        """)

    layer_data = npe2.read([obj_path], stack=False)
    assert isinstance(layer_data, list)
    assert len(layer_data) == 1
    assert isinstance(layer_data[0], tuple)
    assert layer_data[0][2] == 'surface'


@pytest.mark.parametrize(
    ('subject', 'vertices', 'faces'),
    [
        (['#\n'], [], []),
        ([''], [], []),
        (['v 1 -2.5 3'], [[1.0, -2.5, 3.0]], []),
        (['f 1 2 3'], [], [[0, 1, 2]]),
        (['f 1// 2// 3//'], [], [[0, 1, 2]]),
        (['f 1/8/9/ 2/5/6 3/0/0'], [], [[0, 1, 2]]),
        (
            ['v 1 2 3', 'v 3 4 5', 'v 0 0 1', 'f 1 2 3'],
            [[1.0, 2.0, 3.0], [3.0, 4.0, 5.0], [0.0, 0.0, 1.0]],
            [[0, 1, 2]],
        ),
    ],
)
def test_read_wavefront_obj_lines(subject, vertices, faces):
    assert _read_wavefront_obj_lines(subject) == (vertices, faces)


def test_read_obj_with_quads():
    lines = ['v 1 2 3', 'v 3 4 5', 'v 0 0 1', 'v 6 -7 1', 'f 1 2 3 4']
    with pytest.raises(
        ValueError, match='Only triangular faces are supported'
    ):
        _read_wavefront_obj_lines(lines)
