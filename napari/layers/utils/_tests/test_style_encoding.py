from typing import Any, Union

import numpy as np
import pandas as pd
import pytest

from napari.layers.utils._style_encoding import (
    _ConstantStyleEncoding,
    _DerivedStyleEncoding,
    _ManualStyleEncoding,
)
from napari.utils.events.custom_types import Array


@pytest.fixture
def features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            'scalar': [1, 2, 3],
            'vector': [[1, 1], [2, 2], [3, 3]],
        }
    )


Scalar = Array[int, ()]
ScalarArray = Array[int, (-1,)]


class ScalarConstantEncoding(_ConstantStyleEncoding[Scalar, ScalarArray]):
    constant: Scalar


def test_scalar_constant_encoding_update_all(features):
    encoding = ScalarConstantEncoding(constant=0)
    values = encoding._update(features)
    np.testing.assert_array_equal(values, 0)


def test_scalar_constant_encoding_update_some(features):
    encoding = ScalarConstantEncoding(constant=0)
    values = encoding._update(features, indices=[0, 2])
    np.testing.assert_array_equal(values, 0)


def test_scalar_constant_encoding_append():
    encoding = ScalarConstantEncoding(constant=0)
    encoding._append(Vector.validate_type([4, 5]))
    np.testing.assert_array_equal(encoding._values(), 0)


def test_scalar_constant_encoding_delete():
    encoding = ScalarConstantEncoding(constant=0)
    encoding._delete([0, 2])
    np.testing.assert_array_equal(encoding._values(), 0)


def test_scalar_constant_encoding_clear():
    encoding = ScalarConstantEncoding(constant=0)
    encoding._clear()
    np.testing.assert_array_equal(encoding._values(), 0)


class ScalarManualEncoding(_ManualStyleEncoding[Scalar, ScalarArray]):
    array: ScalarArray
    default: Scalar = -1


def test_scalar_manual_encoding_update_with_shorter(features):
    encoding = ScalarManualEncoding(array=[1, 2, 3, 4], default=-1)
    values = encoding._update(features)
    np.testing.assert_array_equal(values, [1, 2, 3])


def test_scalar_manual_encoding_update_with_equal_length(features):
    encoding = ScalarManualEncoding(array=[1, 2, 3], default=-1)
    values = encoding._update(features)
    np.testing.assert_array_equal(values, [1, 2, 3])


def test_scalar_manual_encoding_update_with_longer(features):
    encoding = ScalarManualEncoding(array=[1, 2], default=-1)
    values = encoding._update(features)
    np.testing.assert_array_equal(values, [1, 2, -1])


def test_scalar_manual_encoding_append():
    encoding = ScalarManualEncoding(array=[1, 2, 3])
    encoding._append(Vector.validate_type([4, 5]))
    np.testing.assert_array_equal(encoding._values(), [1, 2, 3, 4, 5])


def test_scalar_manual_encoding_delete():
    encoding = ScalarManualEncoding(array=[1, 2, 3])
    encoding._delete([0, 2])
    np.testing.assert_array_equal(encoding._values(), [2])


def test_scalar_manual_encoding_clear():
    encoding = ScalarManualEncoding(array=[1, 2, 3])
    encoding._clear()
    np.testing.assert_array_equal(encoding._values(), [1, 2, 3])


class ScalarDerivedEncoding(_DerivedStyleEncoding[Scalar, ScalarArray]):
    feature: str
    fallback: Scalar = -1

    def __call__(self, features: Any) -> ScalarArray:
        return ScalarArray.validate_type(features[self.feature])


def test_scalar_derived_encoding_update_all(features):
    encoding = ScalarDerivedEncoding(feature='scalar')
    values = encoding._update(features)
    expected_values = features['scalar']
    np.testing.assert_array_equal(values, expected_values)


def test_scalar_derived_encoding_update_some(features):
    encoding = ScalarDerivedEncoding(feature='scalar')

    values = encoding._update(features, indices=[0, 2])

    expected_values = features['scalar'][[0, 2]]
    np.testing.assert_array_equal(values, expected_values)


def test_scalar_derived_encoding_append():
    encoding = ScalarDerivedEncoding(feature='scalar')
    encoding._cached = ScalarArray.validate_type([1, 2, 3])

    encoding._append(ScalarArray.validate_type([4, 5]))

    np.testing.assert_array_equal(encoding._values(), [1, 2, 3, 4, 5])


def test_scalar_derived_encoding_delete():
    encoding = ScalarDerivedEncoding(feature='scalar')
    encoding._cached = ScalarArray.validate_type([1, 2, 3])

    encoding._delete([0, 2])

    np.testing.assert_array_equal(encoding._values(), [2])


def test_scalar_derived_encoding_clear():
    encoding = ScalarDerivedEncoding(feature='scalar')
    encoding._cached = ScalarArray.validate_type([1, 2, 3])

    encoding._clear()

    np.testing.assert_array_equal(encoding._values(), [])


Vector = Array[int, (2,)]
VectorArray = Array[int, (-1, 2)]


class VectorConstantEncoding(_ConstantStyleEncoding[Vector, VectorArray]):
    constant: Vector


class VectorManualEncoding(_ManualStyleEncoding[Vector, VectorArray]):
    array: VectorArray
    default: Vector


class VectorDerivedEncoding(_DerivedStyleEncoding[Vector, VectorArray]):
    feature: str
    fallback: Vector

    def __call__(self, features: Any) -> Union[Vector, VectorArray]:
        return VectorArray.validate_type(features.self.feature)
