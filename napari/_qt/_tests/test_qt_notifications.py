import sys
import threading
import time
import warnings
from unittest.mock import patch

import dask.array as da
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

PY37_OR_LOWER = sys.version_info[:2] <= (3, 7)


def _threading_warn():
    thr = threading.Thread(target=_warn)
    thr.start()


def _warn():
    time.sleep(0.1)
    warnings.warn('warning!')


def _threading_raise():
    thr = threading.Thread(target=_raise)
    thr.start()


def _raise():
    time.sleep(0.1)
    raise ValueError("error!")


@pytest.fixture
def clean_current(monkeypatch, qtbot):
    from napari._qt.qt_main_window import _QtMainWindow

    def none_return(*_, **__):
        return None

    base_show = NapariQtNotification.show

    def store_widget(self, *args, **kwargs):
        qtbot.addWidget(self)
        base_show(self, *args, **kwargs)

    # monkeypatch.setattr(qt_notification.QPropertyAnimation, "start", none_return)
    monkeypatch.setattr(_QtMainWindow, "current", none_return)
    monkeypatch.setattr(NapariQtNotification, "show", store_widget)


@pytest.mark.parametrize(
    "raise_func,warn_func",
    [(_raise, _warn), (_threading_raise, _threading_warn)],
)
@pytest.mark.order(11)
def test_notification_manager_via_gui(
    qtbot, raise_func, warn_func, clean_current
):
    """
    Test that the notification_manager intercepts `sys.excepthook`` and
    `threading.excepthook`.
    """
    errButton = QPushButton()
    warnButton = QPushButton()
    errButton.clicked.connect(raise_func)
    warnButton.clicked.connect(warn_func)

    with notification_manager:
        for btt, expected_message in [
            (errButton, 'error!'),
            (warnButton, 'warning!'),
        ]:
            notification_manager.records = []
            qtbot.mouseClick(btt, Qt.LeftButton)
            qtbot.wait(500)
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
    from napari.settings import get_settings

    settings = get_settings()

    monkeypatch.delenv('NAPARI_CATCH_ERRORS', raising=False)
    monkeypatch.setattr(
        settings.application,
        'gui_notification_level',
        NotificationSeverity.INFO,
    )
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
def test_notification_error(mock_show, monkeypatch, clean_current):
    from napari.settings import get_settings

    settings = get_settings()

    monkeypatch.delenv('NAPARI_CATCH_ERRORS', raising=False)
    monkeypatch.setattr(
        settings.application,
        'gui_notification_level',
        NotificationSeverity.INFO,
    )
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


@pytest.mark.skipif(PY37_OR_LOWER, reason="Fails on py37")
def test_notifications_error_with_threading(make_napari_viewer):
    """Test notifications of `threading` threads, using a dask example."""
    random_image = da.random.random(size=(50, 50))
    with notification_manager:
        viewer = make_napari_viewer()
        viewer.add_image(random_image)
        result = da.divide(random_image, da.zeros(50, 50))
        viewer.add_image(result)
        assert len(notification_manager.records) >= 1
        notification_manager.records = []
