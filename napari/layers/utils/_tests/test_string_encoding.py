import numpy as np
import pandas as pd
import pytest

from napari.layers.utils.string_encoding import (
    ConstantStringEncoding,
    DirectStringEncoding,
    FormatStringEncoding,
    ManualStringEncoding,
    StringEncoding,
)


def make_features_with_no_columns(*, num_rows) -> pd.DataFrame:
    return pd.DataFrame({}, index=range(num_rows))


@pytest.fixture
def features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            'class': ['a', 'b', 'c'],
            'confidence': [0.5, 1, 0.25],
        }
    )


@pytest.fixture
def numeric_features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            'label': [1, 2, 3],
            'confidence': [0.5, 1, 0.25],
        }
    )


def test_constant_call_with_no_rows():
    features = make_features_with_no_columns(num_rows=0)
    encoding = ConstantStringEncoding(constant='abc')

    values = encoding(features)

    np.testing.assert_equal(values, 'abc')


def test_constant_call_with_some_rows():
    features = make_features_with_no_columns(num_rows=3)
    encoding = ConstantStringEncoding(constant='abc')

    values = encoding(features)

    np.testing.assert_equal(values, 'abc')


def test_manual_call_with_no_rows():
    features = make_features_with_no_columns(num_rows=0)
    array = ['a', 'b', 'c']
    default = 'd'
    encoding = ManualStringEncoding(array=array, default=default)

    values = encoding(features)

    np.testing.assert_array_equal(values, np.array([], dtype=str))


def test_manual_call_with_fewer_rows():
    features = make_features_with_no_columns(num_rows=2)
    array = ['a', 'b', 'c']
    default = 'd'
    encoding = ManualStringEncoding(array=array, default=default)

    values = encoding(features)

    np.testing.assert_array_equal(values, ['a', 'b'])


def test_manual_call_with_same_rows():
    features = make_features_with_no_columns(num_rows=3)
    array = ['a', 'b', 'c']
    default = 'd'
    encoding = ManualStringEncoding(array=array, default=default)

    values = encoding(features)

    np.testing.assert_array_equal(values, ['a', 'b', 'c'])


def test_manual_with_more_rows():
    features = make_features_with_no_columns(num_rows=4)
    array = ['a', 'b', 'c']
    default = 'd'
    encoding = ManualStringEncoding(array=array, default=default)

    values = encoding(features)

    np.testing.assert_array_equal(values, ['a', 'b', 'c', 'd'])


def test_direct(features):
    encoding = DirectStringEncoding(feature='class')
    values = encoding(features)
    np.testing.assert_array_equal(values, features['class'])


def test_direct_with_a_missing_feature(features):
    encoding = DirectStringEncoding(feature='not_class')
    with pytest.raises(KeyError):
        encoding(features)


def test_format(features):
    encoding = FormatStringEncoding(format='{class}: {confidence:.2f}')
    values = encoding(features)
    np.testing.assert_array_equal(values, ['a: 0.50', 'b: 1.00', 'c: 0.25'])


def test_format_with_bad_string(features):
    encoding = FormatStringEncoding(format='{class}: {confidence:.2f')
    with pytest.raises(ValueError):
        encoding(features)


def test_format_with_missing_field(features):
    encoding = FormatStringEncoding(format='{class}: {score:.2f}')
    with pytest.raises(KeyError):
        encoding(features)


def test_format_with_mixed_feature_numeric_types(numeric_features):
    encoding = FormatStringEncoding(format='{label:d}: {confidence:.2f}')
    values = encoding(numeric_features)
    np.testing.assert_array_equal(values, ['1: 0.50', '2: 1.00', '3: 0.25'])


def test_validate_from_format_string():
    argument = '{class}: {score:.2f}'
    expected = FormatStringEncoding(format=argument)

    actual = StringEncoding.validate(argument)

    assert actual == expected


def test_validate_from_non_format_string():
    argument = 'abc'
    expected = DirectStringEncoding(feature=argument)

    actual = StringEncoding.validate(argument)

    assert actual == expected


def test_validate_from_sequence():
    argument = ['a', 'b', 'c']
    expected = ManualStringEncoding(array=argument)

    actual = StringEncoding.validate(argument)

    assert actual == expected


def test_validate_from_constant_dict():
    constant = 'test'
    argument = {'constant': constant}
    expected = ConstantStringEncoding(constant=constant)

    actual = StringEncoding.validate(argument)

    assert actual == expected


def test_validate_from_manual_dict():
    array = ['a', 'b', 'c']
    default = 'd'
    argument = {'array': array, 'default': default}
    expected = ManualStringEncoding(array=array, default=default)

    actual = StringEncoding.validate(argument)

    assert actual == expected


def test_validate_from_direct_dict():
    feature = 'class'
    argument = {'feature': feature}
    expected = DirectStringEncoding(feature=feature)

    actual = StringEncoding.validate(argument)

    assert actual == expected


def test_validate_from_format_dict():
    format = '{class}: {score:.2f}'
    argument = {'format': format}
    expected = FormatStringEncoding(format=format)

    actual = StringEncoding.validate(argument)

    assert actual == expected
