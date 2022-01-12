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
    array = encoding._get_array(features)
    np.testing.assert_equal(array, np.empty((0,), dtype=str))


def test_constant_with_some_rows():
    features = pd.DataFrame({}, index=range(3))
    encoding = ConstantStringEncoding(constant='text')
    array = encoding._get_array(features)
    np.testing.assert_equal(array, np.array(['text'] * 3))


def test_constant_with_some_rows_and_some_indices():
    features = pd.DataFrame({}, index=range(3))
    encoding = ConstantStringEncoding(constant='text')
    array = encoding._get_array(features, indices=[0, 2])
    np.testing.assert_equal(array, np.array(['text'] * 2))


def test_manual_with_same_rows():
    values = ['x', 'y', 'z']
    default = 'w'
    features = pd.DataFrame({}, index=range(3))

    encoding = ManualStringEncoding(array=values, default=default)
    array = encoding._get_array(features)

    np.testing.assert_array_equal(array, values)


def test_manual_with_more_rows():
    values = ['x', 'y', 'z']
    default = 'w'
    features = pd.DataFrame({}, index=range(4))

    encoding = ManualStringEncoding(array=values, default=default)
    array = encoding._get_array(features)

    np.testing.assert_array_equal(array, values + [default])


def test_direct():
    features = pd.DataFrame({'class': ['a', 'b', 'c']})

    encoding = DirectStringEncoding(feature='class')
    array = encoding._get_array(features)

    np.testing.assert_array_equal(array, features['class'])


def test_format():
    features = pd.DataFrame(
        {
            'class': ['a', 'b', 'c'],
            'confidence': [0.5, 1, 0.25],
        }
    )

    encoding = FormatStringEncoding(format_string='{class}: {confidence:.2f}')
    array = encoding._get_array(features)

    np.testing.assert_array_equal(array, ['a: 0.50', 'b: 1.00', 'c: 0.25'])
