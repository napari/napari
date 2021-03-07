import sys

from napari.utils.notifications import notification_manager, show_info


def test_notification_manager_no_gui():
    """
    Direct test of the notification manager.

    This does not test the integration with the gui, but test that the
    notification manager itself can receive a info, warning or error.
    """

    with notification_manager:
        show_info('this is one way of showing an information message')
        notification_manager.receive_info(
            'This is another information message'
        )

        class PurposefulException(Exception):
            pass

        try:
            raise PurposefulException("this is an exception")
        except PurposefulException:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            notification_manager.receive_error(
                exc_type, exc_value, exc_traceback
            )
