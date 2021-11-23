import numpy as np

from napari.layers.utils.string_encoding import (
    ConstantStringEncoding,
    DirectStringEncoding,
    FormatStringEncoding,
    IdentityStringEncoding,
)


def test_constant_with_no_rows():
    encoding = ConstantStringEncoding(constant='test')
    array = encoding._get_array({}, 0)
    np.testing.assert_equal(array, np.empty((0,), dtype=str))


def test_constant_with_some_rows():
    encoding = ConstantStringEncoding(constant='text')
    array = encoding._get_array({}, 3)
    np.testing.assert_equal(array, np.array(['text'] * 3))


def test_constant_with_some_rows_and_some_indices():
    encoding = ConstantStringEncoding(constant='text')
    array = encoding._get_array({}, 3, indices=[0, 2])
    np.testing.assert_equal(array, np.array(['text'] * 2))


def test_direct_with_same_rows():
    values = ['x', 'y', 'z']
    default = 'w'

    encoding = DirectStringEncoding(array=values, default=default)
    array = encoding._get_array({}, len(values))

    np.testing.assert_array_equal(array, values)


def test_direct_with_more_rows():
    values = ['x', 'y', 'z']
    default = 'w'

    encoding = DirectStringEncoding(array=values, default=default)
    array = encoding._get_array({}, len(values) + 1)

    np.testing.assert_array_equal(array, values + [default])


def test_identity():
    n_rows = 3
    class_values = np.array(['a', 'b', 'c'])
    properties = {'class': class_values}

    encoding = IdentityStringEncoding(property='class')
    array = encoding._get_array(properties, n_rows)

    np.testing.assert_array_equal(array, class_values)


def test_format():
    n_rows = 3
    properties = {
        'class': np.array(['a', 'b', 'c']),
        'confidence': np.array([0.5, 1, 0.25]),
    }

    encoding = FormatStringEncoding(format_string='{class}: {confidence:.2f}')
    array = encoding._get_array(properties, n_rows)

    np.testing.assert_array_equal(array, ['a: 0.50', 'b: 1.00', 'c: 0.25'])
