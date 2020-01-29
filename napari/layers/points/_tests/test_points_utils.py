import numpy as np
import pandas as pd

from napari.layers.points.points_utils import (
    dataframe_to_properties,
    guess_continuous,
)


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
