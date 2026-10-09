import pytest

from napari._qt import _check_qt_version


@pytest.mark.parametrize('qt_version', ['6.7.0', '6.7.3', '6.10.1'])
def test_supported_qt_version(qt_version):
    _check_qt_version(qt_version, 'PyQt6')


def test_old_qt6_version():
    with pytest.raises(
        ImportError, match=r'Qt >= 6\.7.*PyQt6 with Qt 6\.5\.3'
    ) as exc:
        _check_qt_version('6.5.3', 'PyQt6')
    msg = str(exc.value)
    assert "pip install -U 'napari[pyqt6]'" in msg
    assert 'QT_API' not in msg


def test_qt5_version():
    with pytest.raises(
        ImportError, match=r'Qt >= 6\.7.*PyQt5 with Qt 5\.15\.2'
    ) as exc:
        _check_qt_version('5.15.2', 'PyQt5')
    msg = str(exc.value)
    assert 'QT_API=pyqt6' in msg
    assert 'uninstall PyQt5' in msg
    assert "pip install -U 'napari[pyqt6]'" in msg
