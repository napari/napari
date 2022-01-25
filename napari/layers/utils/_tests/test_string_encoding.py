import numpy as np
import pandas as pd
import pytest

from napari.layers.utils.string_encoding import (
    ConstantStringEncoding,
    DirectStringEncoding,
    FormatStringEncoding,
    ManualStringEncoding,
)


def make_features_with_no_columns(*, num_rows):
    return pd.DataFrame({}, index=range(num_rows))


def make_features_with_class_confidence_columns():
    return pd.DataFrame(
        {
            'class': ['a', 'b', 'c'],
            'confidence': [0.5, 1, 0.25],
        }
    )


def test_constant_call_with_no_rows():
    features = make_features_with_no_columns(num_rows=0)
    encoding = ConstantStringEncoding(constant='text')

    values = encoding(features)

    np.testing.assert_equal(values, ['text'])


def test_constant_call_with_some_rows():
    features = make_features_with_no_columns(num_rows=3)
    encoding = ConstantStringEncoding(constant='text')

    values = encoding(features)

    np.testing.assert_equal(values, ['text'])


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


def test_direct():
    features = make_features_with_class_confidence_columns()
    encoding = DirectStringEncoding(feature='class')

    values = encoding(features)

    np.testing.assert_array_equal(values, features['class'])


def test_direct_with_a_missing_feature():
    features = make_features_with_class_confidence_columns()
    encoding = DirectStringEncoding(feature='not_class')

    with pytest.raises(KeyError):
        encoding(features)


def test_format():
    features = make_features_with_class_confidence_columns()
    encoding = FormatStringEncoding(format_string='{class}: {confidence:.2f}')

    values = encoding(features)

    np.testing.assert_array_equal(values, ['a: 0.50', 'b: 1.00', 'c: 0.25'])


def test_format_with_bad_string():
    features = make_features_with_class_confidence_columns()
    encoding = FormatStringEncoding(format_string='{class}: {confidence:.2f')

    with pytest.raises(ValueError):
        encoding(features)


def test_format_with_missing_field():
    features = make_features_with_class_confidence_columns()
    encoding = FormatStringEncoding(format_string='{class}: {score:.2f}')

    with pytest.raises(KeyError):
        encoding(features)
