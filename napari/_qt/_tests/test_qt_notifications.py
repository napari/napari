import threading
import warnings
from concurrent.futures import Future
from dataclasses import dataclass
from unittest.mock import MagicMock

import dask.array as da
import pytest
from qtpy.QtCore import Qt, QThread
from qtpy.QtWidgets import QPushButton, QWidget

from napari._qt.dialogs.qt_notification import (
    NapariQtNotification,
    TracebackDialog,
)
from napari._tests.utils import DEFAULT_TIMEOUT_SECS, skip_on_win_ci
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
    mock_window._qt_viewer._welcome_widget = widget

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


@dataclass
class ShowStatus:
    show_notification_count: int = 0
    show_traceback_count: int = 0


@pytest.fixture(autouse=True)
def raise_on_show(monkeypatch, qtbot):
    def raise_prepare(text):
        def _raise_on_call(self, *args, **kwargs):
            raise RuntimeError(text)

        return _raise_on_call

    monkeypatch.setattr(
        NapariQtNotification, 'show', raise_prepare("notification show")
    )
    monkeypatch.setattr(
        TracebackDialog, 'show', raise_prepare("traceback show")
    )
    monkeypatch.setattr(
        NapariQtNotification,
        'close_with_fade',
        raise_prepare("close_with_fade"),
    )


@pytest.fixture
def count_show(monkeypatch, qtbot):

    stat = ShowStatus()

    def mock_show_notif(_):
        stat.show_notification_count += 1

    def mock_show_traceback(_):
        stat.show_traceback_count += 1

    monkeypatch.setattr(NapariQtNotification, "show", mock_show_notif)
    monkeypatch.setattr(TracebackDialog, "show", mock_show_traceback)

    return stat


@pytest.fixture(autouse=True)
def ensure_qtbot(monkeypatch, qtbot):
    old_notif_init = NapariQtNotification.__init__
    old_traceback_init = TracebackDialog.__init__

    def mock_notif_init(self, *args, **kwargs):
        old_notif_init(self, *args, **kwargs)
        qtbot.add_widget(self)

    def mock_traceback_init(self, *args, **kwargs):
        old_traceback_init(self, *args, **kwargs)
        qtbot.add_widget(self)

    monkeypatch.setattr(NapariQtNotification, "__init__", mock_notif_init)
    monkeypatch.setattr(TracebackDialog, "__init__", mock_traceback_init)


def test_clean_current_path_exist(make_napari_viewer):
    """If this test fail then you need to fix also clean_current fixture"""
    assert isinstance(
        make_napari_viewer().window._qt_viewer._welcome_widget, QWidget
    )


@pytest.mark.parametrize(
    "raise_func,warn_func",
    [(_raise, _warn), (_threading_raise, _threading_warn)],
)
def test_notification_manager_via_gui(
    count_show, qtbot, raise_func, warn_func, clean_current, monkeypatch
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
    monkeypatch.setattr(
        NapariQtNotification, "show_notification", lambda x: None
    )
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


def test_show_notification_from_thread(
    count_show, monkeypatch, qtbot, clean_current
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
            assert count_show.show_notification_count == 1

    thread = CustomThread()
    with qtbot.waitSignal(thread.finished):
        thread.start()


@pytest.mark.parametrize('severity', NotificationSeverity.__members__)
def test_notification_display(
    count_show, severity, monkeypatch, clean_current
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
        assert count_show.show_notification_count == 1
    else:
        assert count_show.show_notification_count == 0

    dialog = NapariQtNotification.from_notification(notif)
    assert not dialog.property('expanded')
    dialog.toggle_expansion()
    assert dialog.property('expanded')
    dialog.toggle_expansion()
    assert not dialog.property('expanded')


def test_notification_error(count_show, monkeypatch):
    from napari.settings import get_settings

    settings = get_settings()

    monkeypatch.delenv('NAPARI_CATCH_ERRORS', raising=False)
    monkeypatch.setattr(
        NapariQtNotification, "close_with_fade", lambda x, y: None
    )
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
    assert count_show.show_traceback_count == 0
    bttn.click()
    assert count_show.show_traceback_count == 1


@skip_on_win_ci
@pytest.mark.sync_only
def test_notifications_error_with_threading(
    make_napari_viewer, clean_current, monkeypatch
):
    """Test notifications of `threading` threads, using a dask example."""
    random_image = da.random.random((10, 10))
    monkeypatch.setattr(
        NapariQtNotification, "show_notification", lambda x: None
    )
    with notification_manager:
        viewer = make_napari_viewer(strict_qt=False)
        viewer.add_image(random_image)
        result = da.divide(random_image, da.zeros((10, 10)))
        viewer.add_image(result)
        assert len(notification_manager.records) >= 1
        notification_manager.records = []
