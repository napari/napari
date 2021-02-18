from napari._qt.qt_resources import build_qt_resources


def test_resources():
    """Test that we can build icons and resources."""
    resources = build_qt_resources()
    assert 'from qtpy import QtCore' in resources
