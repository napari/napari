import importlib
from unittest.mock import patch

import numpy as np
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
    ('path', 'closed', 'bevel', 'expeected'),
    [
        ([[0, 0], [0, 10], [10, 10], [10, 0]], True, False, 10),
        ([[0, 0], [0, 10], [10, 10], [10, 0]], False, False, 8),
        ([[0, 0], [0, 10], [10, 10], [10, 0]], True, True, 14),
        ([[0, 0], [0, 10], [10, 10], [10, 0]], False, True, 10),
        ([[2, 10], [0, -5], [-2, 10], [-2, -10], [2, -10]], True, False, 15),
        ([[0, 0], [0, 10]], False, False, 4),
        ([[0, 0], [0, 10], [0, 20]], False, False, 6),
    ],
)
@pytest.mark.usefixtures('_disable_jit')
def test_generate_2D_edge_meshes(path, closed, bevel, expeected):
    centers, offsets, triangles = ac.generate_2D_edge_meshes(
        np.array(path, dtype='float32'), closed=closed, bevel=bevel
    )
    assert centers.shape == offsets.shape
    assert centers.shape[0] == expeected
    assert triangles.shape[0] == expeected - 2
