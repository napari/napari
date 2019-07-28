import os
from napari._qt.qt_console import QtConsole
from napari import gui_qt

os.environ['NAPARI_TEST'] = '1'


def test_console():
    """Test creating the console."""
    with gui_qt():

        console = QtConsole()
        assert console.kernel_client is not None


def test_console_user_variables():
    """Test creating the console with user variables."""
    with gui_qt():

        console = QtConsole({'var': 3})
        assert console.kernel_client is not None
