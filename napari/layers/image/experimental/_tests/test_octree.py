import numpy as np
import pytest

from napari.layers.image.experimental.octree_tile_builder import _combine_tiles


def _square(value):
    return np.array([[value, value], [value, value]])


def test_combine():
    """Test _combine_tiles().

    Combine 1, 2 or 4 tiles into a single bigger one.
    """
    # Create 4 square arrays:
    # 0  1  2  3
    # -----------
    # 00 11 22 33
    # 00 11 22 33
    tiles = [np.array(_square(i)) for i in range(4)]

    with pytest.raises(ValueError):
        _combine_tiles(tiles[0], tiles[1], tiles[2])  # Too few values.

    with pytest.raises(ValueError):
        _combine_tiles(tiles[0], None, None, None, None)  # Too many values.

    # Combine them the 4 major ways:

    # case1: corner
    # 0X
    # XX
    case1 = _combine_tiles(tiles[0], None, None, None)
    assert case1.shape == (2, 2)
    assert (case1 == tiles[0]).all()

    # case2: bottom edge
    # 01
    # XX
    case2 = _combine_tiles(tiles[0], tiles[1], None, None)
    assert case2.shape == (2, 4)
    assert (case2[0:2, 0:2] == tiles[0]).all()
    assert (case2[0:2, 3:5] == tiles[1]).all()

    # case3: right edge
    # 0X
    # 2X
    case3 = _combine_tiles(tiles[0], None, tiles[2], None)
    assert case3.shape == (4, 2)
    assert (case3[0:2, 0:2] == tiles[0]).all()
    assert (case3[3:5, 0:2] == tiles[2]).all()

    # case4: interior
    # 01
    # 23
    case4 = _combine_tiles(tiles[0], tiles[1], tiles[2], tiles[3])
    assert case4.shape == (4, 4)
    assert (case4[0:2, 0:2] == tiles[0]).all()
    assert (case4[0:2, 3:5] == tiles[1]).all()
    assert (case4[3:5, 0:2] == tiles[2]).all()
    assert (case4[3:5, 3:5] == tiles[3]).all()
