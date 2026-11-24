import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    'construct_layer',
    [
        (
            'from napari.layers import Image; '
            'Image(np.zeros((4, 4))); Image(np.zeros((4, 4)))'
        ),
        (
            'from napari.layers import Labels; '
            'Labels(np.zeros((4, 4), dtype=np.uint8)); '
            'Labels(np.zeros((4, 4), dtype=np.uint8))'
        ),
    ],
)
def test_unitless_layer_keeps_pint_registry_lazy(construct_layer):
    code = f"""
import numpy as np
import pint

registry = pint.get_application_registry()
if not isinstance(registry, (pint.LazyRegistry, pint.UnitRegistry)):
    registry = registry.get()
assert isinstance(registry, pint.LazyRegistry)
{construct_layer}
assert isinstance(registry, pint.LazyRegistry)
"""

    subprocess.run([sys.executable, '-c', code], check=True, timeout=30)


def test_explicit_units_initialize_pint_registry():
    code = """
import numpy as np
import pint

from napari.layers import Image

registry = pint.get_application_registry()
if not isinstance(registry, (pint.LazyRegistry, pint.UnitRegistry)):
    registry = registry.get()
assert isinstance(registry, pint.LazyRegistry)
Image(np.zeros((4, 4)), units=('um', 'nm'))
assert isinstance(registry, pint.UnitRegistry)
"""

    subprocess.run([sys.executable, '-c', code], check=True, timeout=30)
