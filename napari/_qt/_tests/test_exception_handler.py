import logging

import pytest
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QPushButton

from napari._qt.exceptions import ExceptionHandler
from napari.utils.notifications import notification_manager


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


def test_notification_manager_via_gui(qtbot, make_napari_viewer):
    """
    Test that exception triggered by button in the UI, propagate to the manager,
    and are displayed in the UI.
    """

    def raise_():
        raise ValueError("error!")

    def warn_():
        import warnings

        warnings.warn("warning!")

    viewer = make_napari_viewer()
    errButton = QPushButton(viewer.window.qt_viewer)
    warnButton = QPushButton(viewer.window.qt_viewer)
    errButton.clicked.connect(raise_)
    warnButton.clicked.connect(warn_)

    with notification_manager:
        for btt, expected_message in [
            (errButton, 'error!'),
            (warnButton, 'warning!'),
        ]:
            assert btt is not None, errButton
            assert len(notification_manager.records) == 0
            qtbot.mouseClick(btt, Qt.LeftButton)
            qtbot.wait(150)
            assert len(notification_manager.records) == 1
            assert notification_manager.records[0].message == expected_message
            notification_manager.records = []
