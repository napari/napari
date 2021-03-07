import sys
import warnings

import pytest

from napari.utils.notifications import notification_manager, show_info


def test_notification_manager_no_gui():
    """
    Direct test of the notification manager.

    This does not test the integration with the gui, but test that the
    notification manager itself can receive a info, warning or error.
    """

    with notification_manager:

        show_info('this is one way of showing an information message')
        assert len(notification_manager.records) == 1
        notification_manager.receive_info(
            'This is another information message'
        )
        assert len(notification_manager.records) == 2

        # test that exceptions that go through sys.excepthook are catalogued

        class PurposefulException(Exception):
            pass

        with pytest.raises(PurposefulException):
            raise PurposefulException("this is an exception")

        # pytest intercepts the error, so we can manually call sys.excepthook
        assert sys.excepthook == notification_manager.receive_error
        sys.excepthook(*sys.exc_info())
        assert len(notification_manager.records) == 3

        # test that warnings that go through showwarning are catalogued
        # again, pytest intercepts this, so just manually trigger:
        assert warnings.showwarning == notification_manager.receive_warning
        warnings.showwarning('this is a warning', UserWarning, '', 0)
        assert len(notification_manager.records) == 4
