import pytest
from qtpy.QtWidgets import QWidget

from napari._qt.dialogs.qt_notification import NapariQtNotification
from napari.utils.notifications import NotificationSeverity


@pytest.fixture(autouse=True)
def _dismiss_after(monkeypatch):
    # the conftest fixture disables the dismiss timer for every test
    monkeypatch.setattr(NapariQtNotification, 'DISMISS_AFTER', 4000)


@pytest.fixture(autouse=True)
def _clean_notification_instances():
    """`_instances` is a class variable, so keep it out of other tests."""
    NapariQtNotification._instances.clear()
    yield
    NapariQtNotification._instances.clear()


@pytest.fixture
def notification_parent(qtbot):
    parent = QWidget()
    qtbot.addWidget(parent)
    return parent


def _show_notification(parent, message):
    for notification in NapariQtNotification._instances:
        # only the newest notification counts down to being dismissed
        notification.timer_stop()
    notification = NapariQtNotification(
        message, severity=NotificationSeverity.WARNING, parent=parent
    )
    notification.show()
    # windows are never active when running offscreen, so `show` returns
    # before starting the dismiss timer; start it as an active window would
    notification.timer_start()
    return notification


def test_closing_notification_starts_dismiss_timer_of_next(
    notification_parent,
):
    """The notification below a closed one should dismiss itself too."""
    first = _show_notification(notification_parent, 'first')
    second = _show_notification(notification_parent, 'second')
    assert not first.timer.isActive()

    second.close()

    assert first.timer.isActive()
    # the conftest dangling timer check requires a stopped timer
    first.timer_stop()


def test_closing_notification_keeps_hovered_notification_open(
    notification_parent, monkeypatch
):
    """A notification under the mouse is dismissed once the mouse leaves."""
    first = _show_notification(notification_parent, 'first')
    second = _show_notification(notification_parent, 'second')
    monkeypatch.setattr(NapariQtNotification, 'underMouse', lambda self: True)

    second.close()

    assert not first.timer.isActive()
