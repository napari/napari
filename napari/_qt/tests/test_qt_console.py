import os
from napari._qt.qt_console import QtConsole
from IPython import get_ipython


def test_console(qtbot):
    """Test creating the console."""
    # Check no shell is present
    shell = get_ipython()
    assert shell == None

    console = QtConsole()
    qtbot.addWidget(console)

    assert console.kernel_client is not None

    # Check shell is really shutdown
    console.shutdown()
    shell = get_ipython()
    assert shell == None


def test_console_user_variables(qtbot):
    """Test creating the console with user variables."""
    # Check no shell is present
    shell = get_ipython()
    assert shell == None

    console = QtConsole({'var': 3})
    qtbot.addWidget(console)

    assert console.kernel_client is not None
    assert 'var' in console.shell.user_ns
    assert console.shell.user_ns['var'] == 3
    console.shutdown()

    # Check shell is really shutdown
    shell = get_ipython()
    assert shell == None


def test_multiple_consoles(qtbot):
    """Test creating multiple consoles."""
    console_a = QtConsole({'var_a': 3})
    qtbot.addWidget(console_a)
    console_b = QtConsole({'var_b': 4})
    qtbot.addWidget(console_b)

    assert console_a.kernel_client is not None
    assert console_b.kernel_client is not None
    assert 'var_a' in console_a.shell.user_ns
    assert 'var_b' in console_a.shell.user_ns
    console_a.shutdown()
    console_b.shutdown()
