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
    encoding = ConstantStringEncoding(constant='test')
    values = encoding(features)
    np.testing.assert_equal(values, np.empty((0,), dtype=str))


def test_constant_with_some_rows():
    features = pd.DataFrame({}, index=range(3))
    encoding = ConstantStringEncoding(constant='text')
    values = encoding(features)
    np.testing.assert_equal(values, np.array(['text'] * 3))


def test_constant_with_some_rows_and_some_indices():
    features = pd.DataFrame({}, index=range(3))
    encoding = ConstantStringEncoding(constant='text')
    values = encoding(features, indices=[0, 2])
    np.testing.assert_equal(values, np.array(['text'] * 2))


def test_manual_with_same_rows():
    array = ['x', 'y', 'z']
    default = 'w'
    features = pd.DataFrame({}, index=range(3))

    encoding = ManualStringEncoding(array=array, default=default)
    values = encoding(features)

    np.testing.assert_array_equal(values, array)


def test_manual_with_more_rows():
    array = ['x', 'y', 'z']
    default = 'w'
    features = pd.DataFrame({}, index=range(4))

    encoding = ManualStringEncoding(array=array, default=default)
    values = encoding(features)

    np.testing.assert_array_equal(values, array + [default])


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
