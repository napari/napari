import logging

import pytest

from napari._qt.exceptions import ExceptionHandler


# caplog fixture comes from pytest
# https://docs.pytest.org/en/stable/logging.html#caplog-fixture
def test_exception_handler_interactive(qtbot, caplog):
    """Test exception handler logs to console and emits a signal."""
    handler = ExceptionHandler(gui_exceptions=False)
    with qtbot.waitSignal(handler.error, timeout=1000):
        with caplog.at_level(logging.ERROR):
            handler.handle(ValueError, ValueError("whoops"), None)
    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.levelname == 'ERROR'
    assert 'Unhandled exception' in record.message
    assert 'ValueError: whoops' in record.message


# caplog fixture comes from pytest
# https://docs.pytest.org/en/stable/logging.html#caplog-fixture
def test_keyboard_interupt_handler(qtbot, capsys):
    handler = ExceptionHandler()
    with pytest.raises(SystemExit):
        handler.handle(KeyboardInterrupt, KeyboardInterrupt(), None)
    assert capsys.readouterr().err == "Closed by KeyboardInterrupt\n"


def test_exception_handler_gui(qtbot, make_napari_viewer):
    """Test exception handler can create a NapariNotification"""
    viewer = make_napari_viewer()
    handler = ExceptionHandler(gui_exceptions=True)
    with qtbot.waitSignal(handler.error, timeout=1000):
        handler.handle(ValueError, ValueError("whoops"), None)
    assert handler.message in viewer.window.qt_viewer.canvas.native.children()
    handler.message.toggle_expansion()
    assert handler.message.property('expanded') is True
    handler.message.toggle_expansion()
    assert handler.message.property('expanded') is False
    handler.message.close()
