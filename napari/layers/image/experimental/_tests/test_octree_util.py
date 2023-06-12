import pytest

from napari.layers.image.experimental.octree_util import (
    linear_index,
    spiral_index,
)


@pytest.mark.parametrize(
    "ranges",
    [
        [(0, 7), (0, 9)],
        [(0, 8), (0, 8)],
        [(0, 8), (0, 9)],
        [(0, 8), (0, 10)],
        [(10, 23), (10, 24)],
        [(21, 38), (2, 15)],
        [(22, 38), (2, 16)],
    ],
)
def test_spiral_index_against_linear(ranges):
    """Test spiral index set and linear index set match"""

    row_range, col_range = ranges
    row_range = range(*row_range)
    col_range = range(*col_range)
    spiral = set(spiral_index(row_range, col_range))
    linear = set(linear_index(row_range, col_range))

    assert spiral == linear
