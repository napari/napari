import warnings
from unittest.mock import patch

import pytest
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QPushButton

from napari._qt.dialogs.qt_notification import NapariQtNotification
from napari.utils.notifications import (
    ErrorNotification,
    Notification,
    NotificationSeverity,
    notification_manager,
)


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
def test_notification_display(mock_show, severity, monkeypatch):
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
    from napari.utils.settings import SETTINGS

    monkeypatch.delenv('NAPARI_CATCH_ERRORS', raising=False)
    monkeypatch.setattr(SETTINGS.application, 'gui_notification_level', 'info')
    notif = Notification('hi', severity, actions=[('click', lambda x: None)])
    NapariQtNotification.show_notification(notif)
    if NotificationSeverity(severity) >= NotificationSeverity.INFO:
        mock_show.assert_called_once()
    else:
        mock_show.assert_not_called()

    dialog = NapariQtNotification.from_notification(notif)
    assert not dialog.property('expanded')
    dialog.toggle_expansion()
    assert dialog.property('expanded')
    dialog.toggle_expansion()
    assert not dialog.property('expanded')
    dialog.close()


@patch('napari._qt.dialogs.qt_notification.QDialog.show')
def test_notification_error(mock_show, monkeypatch):
    from napari.utils.settings import SETTINGS

    monkeypatch.delenv('NAPARI_CATCH_ERRORS', raising=False)
    monkeypatch.setattr(SETTINGS.application, 'gui_notification_level', 'info')
    try:
        raise ValueError('error!')
    except ValueError as e:
        notif = ErrorNotification(e)

    dialog = NapariQtNotification.from_notification(notif)
    bttn = dialog.row2_widget.findChild(QPushButton)
    assert bttn.text() == 'View Traceback'
    mock_show.assert_not_called()
    bttn.click()
    mock_show.assert_called_once()
