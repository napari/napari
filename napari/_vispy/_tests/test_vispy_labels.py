from unittest.mock import patch

import numpy as np
import pytest

from napari._vispy.layers.labels import (
    MAX_LOAD_FACTOR,
    build_textures_from_dict,
    hash2d_get,
    idx_to_2d,
)


@pytest.fixture(scope='session', autouse=True)
def mock_max_texture_size():
    with patch('napari._vispy.layers.labels.MAX_TEXTURE_SIZE', 2**16):
        yield


def test_idx_to_2d():
    assert idx_to_2d(0, (100, 100)) == (0, 0)
    assert idx_to_2d(1, (100, 100)) == (0, 1)
    assert idx_to_2d(101, (100, 100)) == (1, 1)
    assert idx_to_2d(521, (100, 100)) == (5, 21)
    assert idx_to_2d(100 * 100 + 521, (100, 100)) == (5, 21)


def test_build_textures_from_dict():
    keys, values, col_keys, val_keys, collision = build_textures_from_dict(
        {1: (1, 1, 1, 1), 2: (2, 2, 2, 2)}
    )
    assert not collision
    assert keys.shape == (61, 61)
    assert values.shape == (61, 61, 4)
    assert col_keys.shape == (1, 1)
    assert val_keys.shape == (1, 1, 4)
    assert keys[0, 1] == 1
    assert keys[0, 2] == 2
    assert np.array_equiv(values[0, 1], (1, 1, 1, 1))
    assert np.array_equiv(values[0, 2], (2, 2, 2, 2))


def test_build_textures_from_dict_too_many_labels(monkeypatch):
    with pytest.raises(OverflowError):
        build_textures_from_dict(
            {i: (i, i, i, i) for i in range(1001)}, shape=(10, 10)
        )
    monkeypatch.setattr(
        "napari._vispy.layers.labels.PRIME_NUM_TABLE", [[61], [127]]
    )
    with pytest.raises(OverflowError):
        build_textures_from_dict(
            {i: (i, i, i, i) for i in range((251**2) // 2)},
        )


def test_size_of_texture_square():
    count = int(127 * 127 * MAX_LOAD_FACTOR) - 1
    keys, values, *_ = build_textures_from_dict(
        {i: (i, i, i, i) for i in range(count)}
    )
    assert keys.shape == (127, 127)
    assert values.shape == (127, 127, 4)


def test_size_of_texture_rectangle():
    count = int(128 * 128 * MAX_LOAD_FACTOR) + 5
    keys, values, *_ = build_textures_from_dict(
        {i: (i, i, i, i) for i in range(count)}
    )
    assert keys.shape == (251, 127)
    assert values.shape == (251, 127, 4)


def test_build_textures_from_dict_collision():
    keys, values, key_col, val_col, collision = build_textures_from_dict(
        {1: (1, 1, 1, 1), 26: (2, 2, 2, 2), 27: (3, 3, 3, 3)}, shape=(5, 5)
    )
    assert collision
    assert keys.shape == (5, 5)
    assert keys[0, 1] == 1
    assert keys[0, 2] == 27
    assert key_col[26, 0] == 26
    assert np.array_equiv(values[0, 1], (1, 1, 1, 1))
    assert np.array_equiv(val_col[26, 0], (2, 2, 2, 2))
    assert np.array_equiv(values[0, 2], (3, 3, 3, 3))

    assert hash2d_get(1, keys, key_col) == ((0, 1), True)
    assert hash2d_get(26, keys, key_col) == ((26, 0), False)
    assert hash2d_get(27, keys, key_col) == ((0, 2), True)
