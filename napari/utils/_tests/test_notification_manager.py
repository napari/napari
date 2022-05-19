import sys
import threading
import time
import warnings
from typing import List

import pytest

from napari.utils.notifications import (
    Notification,
    notification_manager,
    show_error,
    show_info,
    show_warning,
)


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
    try:
        from napari._qt.dialogs.qt_notification import NapariQtNotification

        monkeypatch.setattr(NapariQtNotification, "DISMISS_AFTER", 0)
    except ModuleNotFoundError:
        pass
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
        time.sleep(0.01)
        warnings.showwarning('this is a warning', UserWarning, '', 0)

    def _raise():
        time.sleep(0.01)
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
        time.sleep(0.02)

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

        for _ in range(100):
            time.sleep(0.01)
            if (
                len(notification_manager.records) == 2
                and store[-1].type == 'warning'
            ):
                break
        else:
            raise AssertionError("Thread notification not received in time")

    # make sure we've restored the threading except hook
    assert threading.excepthook == previous_threading_exhook

    assert all(isinstance(x, Notification) for x in store)
