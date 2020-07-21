import logging

import pytest

from napari._qt.exceptions import ExceptionHandler


# caplog fixture comes from pytest
# https://docs.pytest.org/en/stable/logging.html#caplog-fixture
def test_exception_handler(qtbot, caplog, capsys):
    """Test exception handler logs to console and emits a signal."""
    handler = ExceptionHandler()
    with qtbot.waitSignal(handler.error, timeout=1000):
        with caplog.at_level(logging.ERROR):
            handler.handle(ValueError, ValueError("whoops"), None)
    assert (len(caplog.records)) == 1
    record = caplog.records[0]
    assert record.levelname == 'ERROR'
    assert 'Unhandled exception' in record.message
    assert 'ValueError: whoops' in record.message
    with pytest.raises(SystemExit):
        handler.handle(KeyboardInterrupt, KeyboardInterrupt(), None)
    assert capsys.readouterr().err == "Close by KeyboardInterrupt\n"
