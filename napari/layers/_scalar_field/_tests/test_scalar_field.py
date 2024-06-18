from napari.layers._scalar_field.scalar_field import ScalarFieldBase
from napari.utils._test_utils import (
    validate_all_params_in_docstring,
    validate_kwargs_sorted,
)


def test_docstring():
    validate_all_params_in_docstring(ScalarFieldBase)
    validate_kwargs_sorted(ScalarFieldBase)
