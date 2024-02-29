import numpy as np
import pytest

from napari._vispy.layers.labels import (
    build_textures_from_dict,
)


def test_build_textures_from_dict():
    values = build_textures_from_dict(
        {0: (0, 0, 0, 0), 1: (1, 1, 1, 1), 2: (2, 2, 2, 2)},
        max_size=10,
    )
    assert values.shape == (3, 1, 4)
    assert np.array_equiv(values[1], (1, 1, 1, 1))
    assert np.array_equiv(values[2], (2, 2, 2, 2))


def test_build_textures_from_dict_exc():
    with pytest.raises(ValueError, match='Cannot create a 2D texture'):
        build_textures_from_dict(
            {0: (0, 0, 0, 0), 1: (1, 1, 1, 1), 2: (2, 2, 2, 2)},
            max_size=1,
        )
