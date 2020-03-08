import numpy as np
import pandas as pd

from napari.layers import Points
from napari.layers.points._points_utils import (
    dataframe_to_properties,
    guess_continuous,
)


def create_known_points_layer():
    """Create points layer with known coordinates."""
    data = [[1, 3], [8, 4], [10, 10], [15, 4]]
    first_point = [1, 3]
    known_non_point = [20, 30]
    n_points = len(data)

    layer = Points(data, size=1)
    assert np.all(layer.data == data)
    assert layer.ndim == 2
    assert len(layer.data) == n_points
    assert len(layer.selected_data) == 0

    return layer, n_points, first_point, known_non_point


def test_dataframe_to_properties():
    properties = {'point_type': np.array(['A', 'B'] * 5)}
    properties_df = pd.DataFrame(properties)
    converted_properties = dataframe_to_properties(properties_df)
    np.testing.assert_equal(converted_properties, properties)


def test_guess_continuous():
    continuous_annotation = np.array([1, 2, 3], dtype=np.float32)
    assert guess_continuous(continuous_annotation)

    categorical_annotation_1 = np.array([True, False], dtype=np.bool)
    assert not guess_continuous(categorical_annotation_1)

    categorical_annotation_2 = np.array([1, 2, 3], dtype=np.int)
    assert not guess_continuous(categorical_annotation_2)
