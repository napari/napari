from typing import (
    TYPE_CHECKING,
)

import numpy as np
from pydantic import GetCoreSchemaHandler
from pydantic_core import core_schema

if TYPE_CHECKING:
    from decimal import Decimal

    Number = int | float | Decimal

# In numpy 2, the semantics of the copy argument in np.array changed
# so that copy=False errors if a copy is needed:
# https://numpy.org/devdocs/numpy_2_0_migration_guide.html#adapting-to-changes-in-the-copy-keyword
#
# In numpy 1, copy=False meant that a copy was avoided unless necessary,
# but would not error.
#
# In most usage like this use np.asarray instead, but sometimes we need
# to use some of the unique arguments of np.array (e.g. ndmin).
#
# This solution assumes numpy 1 by default, and switches to the numpy 2
# value for any release of numpy 2 on PyPI (including betas and RCs).
copy_if_needed: bool | None = False
if np.lib.NumpyVersion(np.__version__) >= '2.0.0b1':
    copy_if_needed = None


class Array(np.ndarray):
    def __class_getitem__(cls, t):
        return type('Array', (Array,), {'__dtype__': t})

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source, handler: GetCoreSchemaHandler
    ):
        return core_schema.no_info_after_validator_function(
            cls.validate_type, core_schema.any_schema()
        )

    @classmethod
    def validate_type(cls, val):
        dtype = getattr(cls, '__dtype__', None)
        if isinstance(dtype, tuple):
            dtype, shape = dtype
        else:
            shape = ()

        result = np.array(
            val, dtype=dtype, copy=copy_if_needed, ndmin=len(shape)
        )

        if any(
            (shape[i] != -1 and shape[i] != result.shape[i])
            for i in range(len(shape))
        ):
            result = result.reshape(shape)
        return result
