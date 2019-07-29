import os
from napari._qt.qt_console import QtConsole
from napari import gui_qt

os.environ['NAPARI_TEST'] = '1'


def test_console():
    """Test creating the console."""
    with gui_qt():

        console = QtConsole()
        assert console.kernel_client is not None
        assert 'viewer' in console.shell.user_ns


def test_console_user_variables():
    """Test creating the console with user variables."""
    with gui_qt():

        console = QtConsole({'var': 3})
        assert console.kernel_client is not None
        assert 'var' in console.shell.user_ns
        assert console.shell.user_ns['var'] == 3


def test_multiple_consoles():
    """Test creating multiple consoles."""
    with gui_qt():

        console_a = QtConsole({'var_a': 3})
        console_b = QtConsole({'var_b': 4})

        assert console_a.kernel_client is not None
        assert console_b.kernel_client is not None
        assert 'var_a' in console_a.shell.user_ns
        assert 'var_b' in console_a.shell.user_ns
