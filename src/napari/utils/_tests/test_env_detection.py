import pytest

from napari.utils._env_detection import detect_installed_qt_bindings

qtpy = pytest.importorskip('qtpy')


def test_detect_installed_qt_bindings():
    assert qtpy.API in detect_installed_qt_bindings()
