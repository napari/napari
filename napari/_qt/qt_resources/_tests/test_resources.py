from napari._qt.qt_resources._icons import _compile_napari_resources


def test_resources():
    """Test that we can build icons and resources."""
    resources = _compile_napari_resources()
    assert 'from qtpy import QtCore' in resources
