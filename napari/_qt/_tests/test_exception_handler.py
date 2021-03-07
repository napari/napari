import logging
import warnings
from unittest.mock import patch

import pytest
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QPushButton

from napari._qt.dialogs.qt_notification import NapariQtNotification
from napari._qt.exceptions import ExceptionHandler
from napari.utils.notifications import (
    Notification,
    NotificationSeverity,
    notification_manager,
)


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


def test_notification_manager_via_gui(qtbot):
    """Test that the notification_manager intercepts Qt excepthook."""

    def raise_():
        raise ValueError("error!")

    def warn_():
        warnings.warn("warning!")

    errButton = QPushButton()
    warnButton = QPushButton()
    errButton.clicked.connect(raise_)
    warnButton.clicked.connect(warn_)

    with notification_manager:
        for btt, expected_message in [
            (errButton, 'error!'),
            (warnButton, 'warning!'),
        ]:
            assert len(notification_manager.records) == 0
            qtbot.mouseClick(btt, Qt.LeftButton)
            qtbot.wait(50)
            assert len(notification_manager.records) == 1
            assert notification_manager.records[0].message == expected_message
            notification_manager.records = []


@pytest.mark.parametrize('severity', NotificationSeverity.__members__)
@patch('napari._qt.dialogs.qt_notification.QDialog.show')
def test_notification_display(mock_show, severity):
    """Test that NapariQtNotification can present a Notification event.

    NOTE: in napari.utils._tests.test_notification_manager, we already test
    that the notification manager successfully overrides sys.excepthook,
    and warnings.showwarning... and that it emits an event which is an instance
    of napari.utils.notifications.Notification.

    in `get_app()`, we connect `notification_manager.notification_ready` to
    `NapariQtNotification.show_notification`, so all we have to test here is
    that show_notification is capable of receiving various event types.
    (we don't need to test that )
    """
    notif = Notification('hi', severity, actions=[('click', lambda x: None)])
    NapariQtNotification.show_notification(notif)
    mock_show.assert_called_once()

    dialog = NapariQtNotification.from_notification(notif)
    assert not dialog.property('expanded')
    dialog.toggle_expansion()
    assert dialog.property('expanded')
    dialog.toggle_expansion()
    assert not dialog.property('expanded')
    dialog.close()
