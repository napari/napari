import threading
import warnings
from concurrent.futures import Future
from unittest.mock import MagicMock, patch

import dask.array as da
import pytest
from qtpy.QtCore import Qt, QThread
from qtpy.QtWidgets import QPushButton, QWidget

from napari._qt.dialogs.qt_notification import NapariQtNotification
from napari._tests.utils import DEFAULT_TIMEOUT_SECS
from napari.utils.notifications import (
    ErrorNotification,
    Notification,
    NotificationSeverity,
    notification_manager,
)


def _threading_warn():
    thr = threading.Thread(target=_warn)
    thr.start()
    thr.join(timeout=DEFAULT_TIMEOUT_SECS)


def _warn():
    warnings.warn('warning!')


def _threading_raise():
    thr = threading.Thread(target=_raise)
    thr.start()
    thr.join(timeout=DEFAULT_TIMEOUT_SECS)


def _raise():
    raise ValueError("error!")


@pytest.fixture
def clean_current(monkeypatch, qtbot):
    from napari._qt.qt_main_window import _QtMainWindow

    base_show = NapariQtNotification.show

    widget = QWidget()
    qtbot.addWidget(widget)
    mock_window = MagicMock()
    widget.resized = MagicMock()
    mock_window._qt_viewer._canvas_overlay = widget

    def mock_current_main_window(*_, **__):
        """
        This return mock main window object to ensure that
        notification dialog has parent added to qtbot
        """
        return mock_window

    def store_widget(self, *args, **kwargs):
        base_show(self, *args, **kwargs)

    monkeypatch.setattr(NapariQtNotification, "show", store_widget)
    monkeypatch.setattr(_QtMainWindow, "current", mock_current_main_window)


def test_clean_current_path_exist(make_napari_viewer):
    """If this test fail then you need to fix also clean_current fixture"""
    assert isinstance(
        make_napari_viewer().window._qt_viewer._canvas_overlay, QWidget
    )


@pytest.mark.parametrize(
    "raise_func,warn_func",
    [(_raise, _warn), (_threading_raise, _threading_warn)],
)
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
    qtbot.addWidget(errButton)
    qtbot.addWidget(warnButton)

    with notification_manager:
        for btt, expected_message in [
            (errButton, 'error!'),
            (warnButton, 'warning!'),
        ]:
            notification_manager.records = []
            qtbot.mouseClick(btt, Qt.MouseButton.LeftButton)
            assert len(notification_manager.records) == 1
            assert notification_manager.records[0].message == expected_message
            notification_manager.records = []


@patch('napari._qt.dialogs.qt_notification.QDialog.show')
def test_show_notification_from_thread(
    mock_show, monkeypatch, qtbot, clean_current
):
    from napari.settings import get_settings

    settings = get_settings()

    monkeypatch.setattr(
        settings.application,
        'gui_notification_level',
        NotificationSeverity.INFO,
    )

    class CustomThread(QThread):
        def run(self):
            notif = Notification(
                'hi',
                NotificationSeverity.INFO,
                actions=[('click', lambda x: None)],
            )
            res = NapariQtNotification.show_notification(notif)
            assert isinstance(res, Future)
            assert res.result(timeout=DEFAULT_TIMEOUT_SECS) is None
            mock_show.assert_called_once()

    thread = CustomThread()
    with qtbot.waitSignal(thread.finished):
        thread.start()


@pytest.mark.parametrize('severity', NotificationSeverity.__members__)
@patch('napari._qt.dialogs.qt_notification.QDialog.show')
def test_notification_display(
    mock_show, severity, monkeypatch, qtbot, clean_current
):
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
    qtbot.add_widget(dialog)
    assert not dialog.property('expanded')
    dialog.toggle_expansion()
    assert dialog.property('expanded')
    dialog.toggle_expansion()
    assert not dialog.property('expanded')


@patch('napari._qt.dialogs.qt_notification.TracebackDialog.show')
def test_notification_error(mock_show, monkeypatch, qtbot):
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

    # This test creates TracebackDialog which is not added to qtbot
    # but its parent is same as parent of NapariQtNotification.
    # so create dummy parent allow to not leak widget.
    widget = QWidget()
    qtbot.add_widget(widget)

    dialog = NapariQtNotification.from_notification(notif, parent=widget)

    bttn = dialog.row2_widget.findChild(QPushButton)
    assert bttn.text() == 'View Traceback'
    mock_show.assert_not_called()
    bttn.click()
    mock_show.assert_called_once()


@pytest.mark.sync_only
def test_notifications_error_with_threading(make_napari_viewer, clean_current):
    """Test notifications of `threading` threads, using a dask example."""
    random_image = da.random.random((10, 10))
    with notification_manager:
        viewer = make_napari_viewer(strict_qt=False)
        viewer.add_image(random_image)
        result = da.divide(random_image, da.zeros((10, 10)))
        viewer.add_image(result)
        assert len(notification_manager.records) >= 1
        notification_manager.records = []
