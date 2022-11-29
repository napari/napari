import sys
import threading
import warnings
from contextlib import suppress
from typing import List

import pytest

from napari._tests.utils import DEFAULT_TIMEOUT_SECS
from napari.utils.notifications import (
    Notification,
    notification_manager,
    show_error,
    show_info,
    show_warning,
)


@pytest.fixture(autouse=True)
def disable_notification_dismiss_timer(monkeypatch):
    """
    This fixture disables starting timer for closing notification
    by setting the value of `NapariQtNotification.DISMISS_AFTER` to 0.

    As Qt timer is realised by thread and keep reference to the object,
    without increase of reference counter object could be garbage collected and
    cause segmentation fault error when Qt (C++) code try to access it without
    checking if Python object exists.
    """

    with suppress(ImportError):
        from napari._qt.dialogs.qt_notification import NapariQtNotification

        monkeypatch.setattr(NapariQtNotification, "DISMISS_AFTER", 0)
        monkeypatch.setattr(NapariQtNotification, "FADE_IN_RATE", 0)
        monkeypatch.setattr(NapariQtNotification, "FADE_OUT_RATE", 0)


# capsys fixture comes from pytest
# https://docs.pytest.org/en/stable/logging.html#caplog-fixture
def test_keyboard_interupt_handler(capsys):
    with pytest.raises(SystemExit):
        notification_manager.receive_error(
            KeyboardInterrupt, KeyboardInterrupt(), None
        )


class PurposefulException(Exception):
    pass


def test_notification_repr_has_message():
    assert "='this is the message'" in repr(
        Notification("this is the message")
    )


def test_notification_manager_no_gui(monkeypatch):
    """
    Direct test of the notification manager.

    This does not test the integration with the gui, but test that the
    notification manager itself can receive a info, warning or error.
    """
    previous_exhook = sys.excepthook
    with notification_manager:
        notification_manager.records.clear()
        # save all of the events that get emitted
        store: List[Notification] = []
        notification_manager.notification_ready.connect(store.append)

        show_info('this is one way of showing an information message')
        assert (
            len(notification_manager.records) == 1
        ), notification_manager.records
        assert store[-1].type == 'info'

        notification_manager.receive_info(
            'This is another information message'
        )
        assert len(notification_manager.records) == 2
        assert len(store) == 2
        assert store[-1].type == 'info'

        # test that exceptions that go through sys.excepthook are catalogued

        with pytest.raises(PurposefulException):
            raise PurposefulException("this is an exception")

        # pytest intercepts the error, so we can manually call sys.excepthook
        assert sys.excepthook == notification_manager.receive_error
        try:
            raise ValueError("a")
        except ValueError:
            sys.excepthook(*sys.exc_info())
        assert len(notification_manager.records) == 3
        assert len(store) == 3
        assert store[-1].type == 'error'

        # test that warnings that go through showwarning are catalogued
        # again, pytest intercepts this, so just manually trigger:
        assert warnings.showwarning == notification_manager.receive_warning
        warnings.showwarning('this is a warning', UserWarning, '', 0)
        assert len(notification_manager.records) == 4
        assert store[-1].type == 'warning'

        show_error('This is an error')
        assert len(notification_manager.records) == 5
        assert store[-1].type == 'error'

        show_warning('This is a warning')
        assert len(notification_manager.records) == 6
        assert store[-1].type == 'warning'

    # make sure we've restored the except hook
    assert sys.excepthook == previous_exhook

    assert all(isinstance(x, Notification) for x in store)


def test_notification_manager_no_gui_with_threading():
    """
    Direct test of the notification manager.

    This does not test the integration with the gui, but test that
    exceptions and warnings from threads are correctly captured.
    """

    def _warn():
        warnings.showwarning('this is a warning', UserWarning, '', 0)

    def _raise():
        with pytest.raises(PurposefulException):
            raise PurposefulException("this is an exception")

    previous_threading_exhook = threading.excepthook

    with notification_manager:
        notification_manager.records.clear()
        # save all of the events that get emitted
        store: List[Notification] = []
        notification_manager.notification_ready.connect(store.append)

        # Test exception inside threads
        assert (
            threading.excepthook == notification_manager.receive_thread_error
        )

        exception_thread = threading.Thread(target=_raise)
        exception_thread.start()
        exception_thread.join(timeout=DEFAULT_TIMEOUT_SECS)

        try:
            raise ValueError("a")
        except ValueError:
            threading.excepthook(sys.exc_info())

        assert len(notification_manager.records) == 1
        assert store[-1].type == 'error'

        # Test warning inside threads
        assert warnings.showwarning == notification_manager.receive_warning
        warning_thread = threading.Thread(target=_warn)
        warning_thread.start()
        warning_thread.join(timeout=DEFAULT_TIMEOUT_SECS)

        assert len(notification_manager.records) == 2
        assert store[-1].type == 'warning'

    # make sure we've restored the threading except hook
    assert threading.excepthook == previous_threading_exhook

    assert all(isinstance(x, Notification) for x in store)
