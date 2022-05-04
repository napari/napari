import numpy as np
import pandas as pd
import pytest

from napari.layers.utils._color_encoding import (
    ColorArray,
    ConstantColorEncoding,
    DirectColorEncoding,
    ManualColorEncoding,
    NominalColorEncoding,
    QuantitativeColorEncoding,
    validate_color_encoding,
)


def make_features_with_no_columns(*, num_rows) -> pd.DataFrame:
    return pd.DataFrame({}, index=range(num_rows))


@pytest.fixture
def features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            'class': ['a', 'b', 'c'],
            'confidence': [0.5, 1, 0],
            'custom_colors': ['red', 'green', 'cyan'],
        }
    )


def test_constant_call_with_no_rows():
    features = make_features_with_no_columns(num_rows=0)
    encoding = ConstantColorEncoding(constant='red')

    values = encoding(features)

    assert_colors_equal(values, 'red')


def test_constant_call_with_some_rows():
    features = make_features_with_no_columns(num_rows=3)
    encoding = ConstantColorEncoding(constant='red')

    values = encoding(features)

    assert_colors_equal(values, 'red')


def test_manual_call_with_no_rows():
    features = make_features_with_no_columns(num_rows=0)
    array = ['red', 'green', 'cyan']
    default = 'yellow'
    encoding = ManualColorEncoding(array=array, default=default)

    values = encoding(features)

    assert_colors_equal(values, [])


def test_manual_call_with_fewer_rows():
    features = make_features_with_no_columns(num_rows=2)
    array = ['red', 'green', 'cyan']
    default = 'yellow'
    encoding = ManualColorEncoding(array=array, default=default)

    values = encoding(features)

    assert_colors_equal(values, ['red', 'green'])


def test_manual_call_with_same_rows():
    features = make_features_with_no_columns(num_rows=3)
    array = ['red', 'green', 'cyan']
    default = 'yellow'
    encoding = ManualColorEncoding(array=array, default=default)

    values = encoding(features)

    assert_colors_equal(values, ['red', 'green', 'cyan'])


def test_manual_with_more_rows():
    features = make_features_with_no_columns(num_rows=4)
    array = ['red', 'green', 'cyan']
    default = 'yellow'
    encoding = ManualColorEncoding(array=array, default=default)

    values = encoding(features)

    assert_colors_equal(values, ['red', 'green', 'cyan', 'yellow'])


def test_direct(features):
    encoding = DirectColorEncoding(feature='custom_colors')
    values = encoding(features)
    assert_colors_equal(values, list(features['custom_colors']))


def test_direct_with_missing_feature(features):
    encoding = DirectColorEncoding(feature='not_class')
    with pytest.raises(KeyError):
        encoding(features)


def test_nominal_with_dict_colormap(features):
    colormap = {'a': 'red', 'b': 'yellow', 'c': 'green'}
    encoding = NominalColorEncoding(
        feature='class',
        colormap=colormap,
    )

    values = encoding(features)

    assert_colors_equal(values, ['red', 'yellow', 'green'])


def test_nominal_with_dict_cycle(features):
    colormap = ['red', 'yellow', 'green']
    encoding = NominalColorEncoding(
        feature='class',
        colormap=colormap,
    )

    values = encoding(features)

    assert_colors_equal(values, ['red', 'yellow', 'green'])


def test_nominal_with_missing_feature(features):
    colormap = {'a': 'red', 'b': 'yellow', 'c': 'green'}
    encoding = NominalColorEncoding(feature='not_class', colormap=colormap)
    with pytest.raises(KeyError):
        encoding(features)


def test_quantitative_with_colormap_name(features):
    colormap = 'gray'
    encoding = QuantitativeColorEncoding(
        feature='confidence', colormap=colormap
    )

    values = encoding(features)

    assert_colors_equal(values, [[c] * 3 for c in features['confidence']])


def test_quantitative_with_colormap_values(features):
    colormap = ['black', 'red']
    encoding = QuantitativeColorEncoding(
        feature='confidence', colormap=colormap
    )
    values = encoding(features)
    assert_colors_equal(values, [[c, 0, 0] for c in features['confidence']])


def test_quantitative_with_contrast_limits(features):
    colormap = 'gray'
    encoding = QuantitativeColorEncoding(
        feature='confidence',
        colormap=colormap,
        contrast_limits=(0, 2),
    )

    values = encoding(features)

    assert encoding.contrast_limits == (0, 2)
    assert_colors_equal(values, [[c / 2] * 3 for c in features['confidence']])


def test_quantitative_with_missing_feature(features):
    colormap = 'gray'
    encoding = QuantitativeColorEncoding(
        feature='not_confidence', colormap=colormap
    )

    with pytest.raises(KeyError):
        encoding(features)


def test_validate_from_string():
    argument = 'class'
    expected = DirectColorEncoding(feature=argument)

    actual = validate_color_encoding(argument)

    assert actual == expected


def test_validate_from_sequence():
    argument = ['red', 'green', 'cyan']
    expected = ManualColorEncoding(array=argument)

    actual = validate_color_encoding(argument)

    assert actual == expected


def test_validate_from_constant_dict():
    constant = 'yellow'
    argument = {'constant': constant}
    expected = ConstantColorEncoding(constant=constant)

    actual = validate_color_encoding(argument)

    assert actual == expected


def test_validate_from_manual_dict():
    array = ['red', 'green', 'cyan']
    default = 'yellow'
    argument = {'array': array, 'default': default}
    expected = ManualColorEncoding(array=array, default=default)

    actual = validate_color_encoding(argument)

    assert actual == expected


def test_validate_from_direct_dict():
    feature = 'class'
    argument = {'feature': feature}
    expected = DirectColorEncoding(feature=feature)

    actual = validate_color_encoding(argument)

    assert actual == expected


def test_validate_from_nominal_dict():
    feature = 'class'
    colormap = ['red', 'green', 'cyan']
    argument = {'feature': feature, 'colormap': colormap}
    expected = NominalColorEncoding(
        feature=feature,
        colormap=colormap,
    )

    actual = validate_color_encoding(argument)

    assert actual == expected


def test_validate_from_quantitative_dict(features):
    feature = 'confidence'
    colormap = 'gray'
    contrast_limits = (0, 2)
    argument = {
        'feature': feature,
        'colormap': colormap,
        'contrast_limits': contrast_limits,
    }
    expected = QuantitativeColorEncoding(
        feature=feature,
        colormap=colormap,
        contrast_limits=contrast_limits,
    )

    actual = validate_color_encoding(argument)

    assert actual == expected


def assert_colors_equal(actual, expected):
    actual_array = ColorArray.validate_type(actual)
    expected_array = ColorArray.validate_type(expected)
    np.testing.assert_array_equal(actual_array, expected_array)
