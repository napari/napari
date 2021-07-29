import napari._qt.qt_resources._icons
from napari._qt.qt_resources._icons import (
    _compile_napari_resources,
    _register_napari_resources,
)


def test_resources():
    """Test that we can build icons and resources."""
    resources = _compile_napari_resources()
    assert 'from qtpy import QtCore' in resources


def test_resources_cleanup():
    """Make sure that resource cleanup is present.."""
    resources = _compile_napari_resources()
    assert "qCleanupResources" in resources

    exec(resources, globals())
    assert "qCleanupResources" in globals()


def test_register_resources(qtbot):
    """Test that resource cleanup is being set properly."""
    _register_napari_resources(False, True)
    # it is updated once resource file is loaded
    assert napari._qt.qt_resources._icons._clear_resources is not None
    assert callable(napari._qt.qt_resources._icons._clear_resources)
