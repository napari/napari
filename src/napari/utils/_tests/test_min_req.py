"""For min req add a simple test to ensure that at least numpy is installed with the proper version.

It is in napari.utils to fail test fast.
"""

import os
import tomllib
from importlib.metadata import version

import pytest

if os.environ.get('MIN_REQ') != '1':
    pytest.skip('Not a min_req run', allow_module_level=True)

PYPROJECT_TOML = os.path.join(
    os.path.dirname(__file__), '..', '..', '..', '..', 'pyproject.toml'
)


def test_numpy_version():
    with open(PYPROJECT_TOML, 'rb') as f:
        pyproject = tomllib.load(f)
    min_req = pyproject['project']['dependencies']
    numpy_version = version('numpy')
    numpy_str = next(req for req in min_req if 'numpy' in req)
    assert numpy_version == numpy_str.split('>=')[1], (
        f'numpy version mismatch: expected {numpy_str.split(">=")[1]}, got {numpy_version}'
    )
