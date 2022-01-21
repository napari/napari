import numpy as np
import pandas as pd

from napari.layers.utils.string_encoding import (
    ConstantStringEncoding,
    DirectStringEncoding,
    FormatStringEncoding,
    ManualStringEncoding,
)


def test_constant_with_no_rows():
    features = pd.DataFrame({}, index=range(0))
    encoding = ConstantStringEncoding(constant='text')

    values = encoding(features)

    np.testing.assert_equal(values, ['text'])


def test_constant_with_some_rows():
    features = pd.DataFrame({}, index=range(3))
    encoding = ConstantStringEncoding(constant='text')

    values = encoding(features)

    np.testing.assert_equal(values, ['text'])


def test_manual_with_fewer_rows():
    array = ['a', 'b', 'c']
    default = 'd'
    features = pd.DataFrame({}, index=range(2))
    encoding = ManualStringEncoding(array=array, default=default)

    values = encoding(features)

    np.testing.assert_array_equal(values, ['a', 'b'])


def test_manual_with_same_rows():
    array = ['a', 'b', 'c']
    default = 'd'
    features = pd.DataFrame({}, index=range(3))
    encoding = ManualStringEncoding(array=array, default=default)

    values = encoding(features)

    np.testing.assert_array_equal(values, ['a', 'b', 'c'])


def test_manual_with_more_rows():
    array = ['a', 'b', 'c']
    default = 'd'
    features = pd.DataFrame({}, index=range(4))
    encoding = ManualStringEncoding(array=array, default=default)

    values = encoding(features)

    np.testing.assert_array_equal(values, ['a', 'b', 'c', 'd'])


def test_direct():
    features = pd.DataFrame({'class': ['a', 'b', 'c']})
    encoding = DirectStringEncoding(feature='class')

    values = encoding(features)

    np.testing.assert_array_equal(values, features['class'])


def test_format():
    features = pd.DataFrame(
        {
            'class': ['a', 'b', 'c'],
            'confidence': [0.5, 1, 0.25],
        }
    )
    encoding = FormatStringEncoding(format_string='{class}: {confidence:.2f}')

    values = encoding(features)

    np.testing.assert_array_equal(values, ['a: 0.50', 'b: 1.00', 'c: 0.25'])
