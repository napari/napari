from itertools import product
from unittest.mock import patch

import numpy as np
import pytest

from napari._vispy.layers.labels import (
    MAX_LOAD_FACTOR,
    PRIME_NUM_TABLE,
    build_textures_from_dict,
    hash2d_get,
    idx_to_2d,
)


@pytest.fixture(scope='module', autouse=True)
def mock_max_texture_size():
    """When running tests in this file, pretend max texture size is 2^16."""
    with patch('napari._vispy.layers.labels.MAX_TEXTURE_SIZE', 2**16):
        yield


def test_idx_to_2d():
    assert idx_to_2d(0, (100, 100)) == (0, 0)
    assert idx_to_2d(1, (100, 100)) == (0, 1)
    assert idx_to_2d(101, (100, 100)) == (1, 1)
    assert idx_to_2d(521, (100, 100)) == (5, 21)
    assert idx_to_2d(100 * 100 + 521, (100, 100)) == (5, 21)


def test_build_textures_from_dict():
    keys, values, collision = build_textures_from_dict(
        {1: (1, 1, 1, 1), 2: (2, 2, 2, 2)}
    )
    assert not collision
    assert keys.shape == (37, 37)
    assert values.shape == (37, 37, 4)
    assert keys[0, 1] == 1
    assert keys[0, 2] == 2
    assert np.array_equiv(values[0, 1], (1, 1, 1, 1))
    assert np.array_equiv(values[0, 2], (2, 2, 2, 2))


def test_build_textures_from_dict_too_many_labels(monkeypatch):
    with pytest.raises(MemoryError):
        build_textures_from_dict(
            {i: (i, i, i, i) for i in range(1001)}, shape=(10, 10)
        )
    monkeypatch.setattr(
        "napari._vispy.layers.labels.PRIME_NUM_TABLE", [[61], [127]]
    )
    with pytest.raises(MemoryError):
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
    keys, values, collision = build_textures_from_dict(
        {1: (1, 1, 1, 1), 26: (2, 2, 2, 2), 27: (3, 3, 3, 3)}, shape=(5, 5)
    )
    assert collision
    assert keys.shape == (5, 5)
    assert keys[0, 1] == 1
    assert keys[0, 2] == 26
    assert keys[0, 3] == 27
    assert np.array_equiv(values[0, 1], (1, 1, 1, 1))
    assert np.array_equiv(values[0, 2], (2, 2, 2, 2))
    assert np.array_equiv(values[0, 3], (3, 3, 3, 3))

    assert hash2d_get(1, keys) == (0, 1)
    assert hash2d_get(26, keys) == (0, 2)
    assert hash2d_get(27, keys) == (0, 3)


def test_collide_keys():
    base_keys = [x * y for x, y in product(PRIME_NUM_TABLE[0], repeat=2)]
    colors = {0: (0, 0, 0, 0), 1: (1, 1, 1, 1)}
    colors.update({i + 10: (1, 0, 0, 1) for i in base_keys})
    colors.update({2 * i + 10: (0, 1, 0, 1) for i in base_keys})
    keys, values, collision = build_textures_from_dict(colors)
    assert not collision
    assert keys.shape == (37, 61)
    assert values.shape == (37, 61, 4)


def test_collide_keys2():
    base_keys = [x * y for x, y in product(PRIME_NUM_TABLE[0], repeat=2)] + [
        x * y for x, y in product(PRIME_NUM_TABLE[0], PRIME_NUM_TABLE[1])
    ]
    colors = {0: (0, 0, 0, 0), 1: (1, 1, 1, 1)}
    colors.update({i + 10: (1, 0, 0, 1) for i in base_keys})
    colors.update({2 * i + 10: (0, 1, 0, 1) for i in base_keys})

    # enforce collision for collision table of size 31
    colors.update({31 * i + 10: (0, 0, 1, 1) for i in base_keys})
    # enforce collision for collision table of size 29
    colors.update({29 * i + 10: (0, 0, 1, 1) for i in base_keys})

    keys, values, collision = build_textures_from_dict(colors)
    assert collision
    assert keys.shape == (37, 37)
    assert values.shape == (37, 37, 4)
