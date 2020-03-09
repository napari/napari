from napari.resources import build_pyqt_resources
import os


def test_resources():
    """Test that we can build icons and resources."""
    temp = os.path.join(os.path.dirname(__file__), '_test_qt.py')
    build_pyqt_resources(out_path=temp, overwrite=True)
    from . import _test_qt

    assert _test_qt.QtCore.__package__ == 'qtpy'
    os.remove(temp)
