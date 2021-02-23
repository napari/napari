import logging
import os
import sys

import pytest
from qtpy.QtCore import Qt

from napari._qt.exceptions import ExceptionHandler
from napari.utils.notifications import notification_manager, show_info


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
    Test that exception trigered by button in the UI, propagate to the manager,
    and are displayed in the UI.
    """

    nothing = object()
    orig = os.environ.get('DEBUG_NAPARI_NOTIFICATION', nothing)
    os.environ['DEBUG_NAPARI_NOTIFICATION'] = '1'
    try:
        viewer = make_napari_viewer()
        qtv = viewer.window.qt_viewer
        errButton = qtv.layerButtons.newErrorButton
        warnButton = qtv.layerButtons.newWarnButton
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
                assert (
                    notification_manager.records[0].message == expected_message
                )
                notification_manager.records = []
    finally:
        if orig is nothing:
            del os.environ['DEBUG_NAPARI_NOTIFICATION']
        else:
            os.environ['DEBUG_NAPARI_NOTIFICATION'] = nothing


def test_notification_manager_no_gui(qtbot, make_napari_viewer):
    """
    Test that exception trigered by button in the UI, propagate to the manager,
    and are displayed in the UI.
    """

    with notification_manager:
        show_info('this is one way of showing an informatin message')
        notification_manager.receive_info(
            'This is another information message'
        )

        class PurposefullException(Exception):
            pass

        try:
            raise PurposefullException("this is an exception")
        except PurposefullException:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            notification_manager.receive_error(
                exc_type, exc_value, exc_traceback
            )
