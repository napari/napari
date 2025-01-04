import importlib
from unittest.mock import patch

import numpy as np
import numpy.testing as npt
import pytest

ac = pytest.importorskip('napari.layers.shapes._accelerated_triangulate')


@pytest.fixture
def _disable_jit(monkeypatch):
    """Fixture to temporarily disable numba JIT during testing.

    This helps to measure coverage and in debugging. *However*, reloading a
    module can cause issues with object instance / class relationships, so
    the `_accelerated_cmap` module should be as small as possible and contain
    no class definitions, only functions.
    """
    pytest.importorskip('numba')
    with patch('numba.core.config.DISABLE_JIT', True):
        importlib.reload(ac)
        yield
    importlib.reload(ac)


@pytest.mark.parametrize(
    ('path', 'closed', 'bevel', 'expected'),
    [
        ([[0, 0], [0, 10], [10, 10], [10, 0]], True, False, 10),
        ([[0, 0], [0, 10], [10, 10], [10, 0]], False, False, 8),
        ([[0, 0], [0, 10], [10, 10], [10, 0]], True, True, 14),
        ([[0, 0], [0, 10], [10, 10], [10, 0]], False, True, 10),
        ([[2, 10], [0, -5], [-2, 10], [-2, -10], [2, -10]], True, False, 15),
        ([[0, 0], [0, 10]], False, False, 4),
        ([[0, 0], [0, 10], [0, 20]], False, False, 6),
        ([[0, 0], [0, 2], [10, 1]], True, False, 9),
        ([[0, 0], [10, 1], [9, 1.1]], False, False, 7),
        ([[9, 0.9], [10, 1], [0, 2]], False, False, 7),
        ([[0, 0], [-10, 1], [-9, 1.1]], False, False, 7),
        ([[-9, 0.9], [-10, 1], [0, 2]], False, False, 7),
    ],
)
@pytest.mark.usefixtures('_disable_jit')
def test_generate_2D_edge_meshes(path, closed, bevel, expected):
    centers, offsets, triangles = ac.generate_2D_edge_meshes(
        np.array(path, dtype='float32'), closed=closed, bevel=bevel
    )
    assert centers.shape == offsets.shape
    assert centers.shape[0] == expected
    assert triangles.shape[0] == expected - 2


@pytest.mark.parametrize(
    ('data', 'expected', 'closed'),
    [
        (
            np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype='float32'),
            np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype='float32'),
            True,
        ),
        (
            np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype='float32'),
            np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype='float32'),
            False,
        ),
        (
            np.array(
                [[0, 0], [1, 0], [1, 0], [1, 1], [0, 1]], dtype='float32'
            ),
            np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype='float32'),
            True,
        ),
        (
            np.array(
                [[0, 0], [1, 0], [1, 0], [1, 1], [0, 1]], dtype='float32'
            ),
            np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype='float32'),
            False,
        ),
        (
            np.array(
                [[0, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 1], [0, 1]],
                dtype='float32',
            ),
            np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype='float32'),
            False,
        ),
        (
            np.array(
                [[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]], dtype='float32'
            ),
            np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype='float32'),
            True,
        ),
        (
            np.array(
                [
                    [0, 0],
                    [1, 0],
                    [1, 1],
                    [0, 1],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                ],
                dtype='float32',
            ),
            np.array(
                [[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]], dtype='float32'
            ),
            True,
        ),
    ],
)
@pytest.mark.usefixtures('_disable_jit')
def test_remove_path_duplicates(data, expected, closed):
    result = ac.remove_path_duplicates(data, closed=closed)
    assert np.all(result == expected)


@pytest.mark.usefixtures('_disable_jit')
def test_create_box_from_bounding():
    bounding = np.array([[0, 0], [2, 2]], dtype='float32')
    box = ac.create_box_from_bounding(bounding)
    assert box.shape == (9, 2)
    npt.assert_array_equal(
        box,
        [
            [0, 0],
            [1, 0],
            [2, 0],
            [2, 1],
            [2, 2],
            [1, 2],
            [0, 2],
            [0, 1],
            [1, 1],
        ],
    )
