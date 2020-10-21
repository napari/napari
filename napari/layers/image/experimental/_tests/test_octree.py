import numpy as np
import pytest

from napari.layers.image.experimental.octree import _combine_tiles


def _square(value):
    return np.array([[value, value], [value, value]])


def test_combine():
    tiles = [np.array(_square(i)) for i in range(4)]

    with pytest.raises(ValueError):
        _combine_tiles(tiles[0])  # Too few values.

    with pytest.raises(ValueError):
        _combine_tiles(tiles[0], None, None, None, None)  # Too many values.

    # 0X
    # XX
    case1 = _combine_tiles(tiles[0], None, None, None)
    assert (case1 == tiles[0]).all()

    # 01
    # XX
    case2 = _combine_tiles(tiles[0], tiles[1], None, None)
    assert (case2[0:2, 0:2] == tiles[0]).all()
    assert (case2[0:2, 3:5] == tiles[1]).all()

    # 0X
    # 2X
    case3 = _combine_tiles(tiles[0], None, tiles[2], None)
    assert (case3[0:2, 0:2] == tiles[0]).all()
    assert (case3[3:5, 0:2] == tiles[2]).all()

    # 01
    # 23
    case4 = _combine_tiles(tiles[0], tiles[1], tiles[2], tiles[3])
    assert (case4[0:2, 0:2] == tiles[0]).all()
    assert (case4[0:2, 3:5] == tiles[1]).all()
    assert (case4[3:5, 0:2] == tiles[2]).all()
    assert (case4[3:5, 3:5] == tiles[3]).all()
