import numpy as np
import pytest

from napari._vispy.layers.labels import (
    build_textures_from_dict,
    hash2d_get,
    idx_to_2D,
)


def test_idx_to_2D():
    assert idx_to_2D(0, (100, 100)) == (0, 0)
    assert idx_to_2D(1, (100, 100)) == (0, 1)
    assert idx_to_2D(101, (100, 100)) == (1, 1)
    assert idx_to_2D(521, (100, 100)) == (5, 21)
    assert idx_to_2D(100 * 100 + 521, (100, 100)) == (5, 21)


def test_build_textures_from_dict():
    keys, values = build_textures_from_dict({1: (1, 1, 1, 1), 2: (2, 2, 2, 2)})
    assert keys.shape == (1000, 1000)
    assert values.shape == (1000, 1000, 4)
    assert keys[0, 1] == 1
    assert keys[0, 2] == 2
    assert np.array_equiv(values[0, 1], (1, 1, 1, 1))
    assert np.array_equiv(values[0, 2], (2, 2, 2, 2))


def test_build_textures_from_dict_too_many_labels():
    with pytest.raises(OverflowError):
        build_textures_from_dict(
            {i: (i, i, i, i) for i in range(1001)}, shape=(10, 10)
        )


def test_build_textures_from_dict_collision():
    keys, values = build_textures_from_dict(
        {1: (1, 1, 1, 1), 26: (2, 2, 2, 2), 27: (3, 3, 3, 3)}, shape=(5, 5)
    )
    assert keys.shape == (5, 5)
    assert keys[0, 1] == 1
    assert keys[0, 2] == 26
    assert keys[0, 3] == 27
    assert np.array_equiv(values[0, 1], (1, 1, 1, 1))
    assert np.array_equiv(values[0, 2], (2, 2, 2, 2))
    assert np.array_equiv(values[0, 3], (3, 3, 3, 3))

    assert hash2d_get(1, keys, values) == (0, 1)
    assert hash2d_get(26, keys, values) == (0, 2)
    assert hash2d_get(27, keys, values) == (0, 3)
