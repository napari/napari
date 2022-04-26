import pytest

from napari.utils import updates


def test_get_napari_pypi_versions():
    versions = updates._get_napari_pypi_versions()
    assert '0.4.15' in versions
    assert len(versions) > 0


def test_get_napari_conda_versions():
    versions = updates._get_napari_pypi_versions()
    assert len(versions) > 0
    assert '0.4.15' in versions


def test_get_installed_versions():
    pass


@pytest.mark.parametrize(
    "test_input,expected",
    [
        ("0.4", True),
        ("0.4.15", True),
        ("0.4.15rc1", False),
        ("0.4.15dev0", False),
        ("0.4.15beta", False),
        ("0.4.15alfa", False),
        (('0', '4'), True),
        (('0', '4', '15'), True),
        (('0', '4', '15', 'rc1'), False),
        (('0', '4', '15', 'beta'), False),
    ],
)
def test_is_stable_version(test_input, expected):
    assert updates._is_stable_version(test_input) == expected


def test_check_updates():
    pass
