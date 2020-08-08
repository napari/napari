import logging
import sys

from napari._qt.exceptions import ExceptionHandler, session_is_interactive


# caplog fixture comes from pytest
# https://docs.pytest.org/en/stable/logging.html#caplog-fixture
def test_exception_handler_interactive(qtbot, caplog):
    """Test exception handler logs to console and emits a signal."""
    _ps1 = getattr(sys, 'ps1', None)
    sys.ps1 = '>>'
    handler = ExceptionHandler()
    with qtbot.waitSignal(handler.error, timeout=1000):
        with caplog.at_level(logging.ERROR):
            handler.handle(ValueError, ValueError("whoops"), None)
    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.levelname == 'ERROR'
    assert 'Unhandled exception' in record.message
    assert 'ValueError: whoops' in record.message
    if _ps1:
        sys.ps1 = _ps1
    else:
        del sys.ps1


def test_exception_handler_gui(qtbot, make_test_viewer):
    """Test exception handler logs to console and emits a signal."""
    viewer = make_test_viewer()
    handler = ExceptionHandler()
    assert not session_is_interactive()
    with qtbot.waitSignal(handler.error, timeout=1000):
        handler.handle(ValueError, ValueError("whoops"), None)
    assert handler.message in viewer.window.qt_viewer.canvas.native.children()
    handler.message.toggle_expansion()
    handler.message.toggle_expansion()
    handler.message.close()
