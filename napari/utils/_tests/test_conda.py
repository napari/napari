import sys
from unittest.mock import patch

import pytest

from napari.utils.conda import is_conda_package


@pytest.mark.parametrize(
    "pkg_name,expected",
    [
        ("some-package", True),
        ("some-other-package", False),
        ("some-package-other", False),
        ("other-some-package", False),
        ("package", False),
        ("some", False),
    ],
)
def test_is_conda_package(pkg_name, expected, tmp_path):
    mocked_conda_meta = tmp_path / 'conda-meta'
    mocked_conda_meta.mkdir()
    mocked_package = mocked_conda_meta / 'some-package-0.1.1-0.json'
    mocked_package.touch()

    with patch.object(sys, 'prefix', tmp_path):
        assert is_conda_package(pkg_name) is expected
