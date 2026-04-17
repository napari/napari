import numpy as np
import pytest

from napari.layers import Shapes
from napari_builtins._measure_shapes import (
    toggle_shape_measures,
    update_features_with_measures,
)


@pytest.mark.parametrize(
    ('shape_type', 'shape_data', 'perimeter', 'area'),
    [
        ('line', [[0, 0], [1, 1]], np.sqrt(2), 0),
        ('path', [[0, 0], [0, 1], [1, 1], [1, 0]], 3, 0),
        ('polygon', [[0, 0], [0, 1], [1, 1], [1, 0]], 4, 1),
        ('rectangle', [[0, 0], [1, 1]], 4, 1),
        ('ellipse', [[0, 0], [1, 1]], np.pi * 2, np.pi),
        ('rectangle', [[0, 0], [1, 2]], 6, 2),
        # no closed formula for ellipse perimeter!
        ('ellipse', [[0, 0], [1, 2]], 9.688447, np.pi * 2),
        # 3d shapes, non axis-aligned
        ('path', [[0, 0, 0], [1, 1, 1]], np.sqrt(3), 0),
        (
            'polygon',
            [[0, 0, 0], [1, 0, 0], [1, 1, 1], [0, 1, 1]],
            2 + 2 * np.sqrt(2),
            np.sqrt(2),
        ),
    ],
)
def test_measure_shapes(shape_type, shape_data, perimeter, area):
    layer = Shapes([shape_data], shape_type=shape_type)
    update_features_with_measures(layer)
    p, a = layer.features.loc[0, ['_perimeter', '_area']]
    np.testing.assert_almost_equal(p, perimeter, decimal=6)
    np.testing.assert_almost_equal(a, area, decimal=6)


def test_toggle_measures():
    layer = Shapes([[0, 0], [0, 1]], shape_type='line')

    toggle_shape_measures(layer)
    np.testing.assert_array_equal(
        layer.features.loc[0, ['_perimeter', '_area']], (1, 0)
    )
    assert layer.text.values[0] == 'P = 1\nA = 0'

    toggle_shape_measures(layer)
    assert '_perimeter' not in layer.features.columns
    assert layer.text.values == ''
