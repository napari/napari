from qtpy.QtCore import Qt

from napari._qt.dialogs.qt_notification import NapariQtNotification
from napari.utils.notifications import Notification


def test_show_while_window_inactive_configures_timer_without_starting_it(
    qt_viewer_, monkeypatch
):
    """show() must set up the dismiss timer even when the window is not
    active, leaving it stopped, so that the timer can be started once the
    window regains focus.
    """
    # the autouse `_disable_notification_dismiss_timer` fixture sets
    # DISMISS_AFTER to 0; override it so the timer can actually run
    monkeypatch.setattr(NapariQtNotification, 'DISMISS_AFTER', 200)

    notif = NapariQtNotification.from_notification(
        Notification('hi'), qt_viewer_.canvas.native
    )
    notif.show()

    assert notif.timer.interval() == 200
    assert notif.timer.isSingleShot()
    assert not notif.timer.isActive()

    # as `_focus_changed` does once the window regains focus
    notif.timer_start()
    assert notif.timer.isActive()
    notif.close()


def test_focus_loss_stops_timers_and_focus_gain_restarts_top_timer(
    make_napari_viewer, monkeypatch, qapp
):
    """Focus loss stops all timers; focus gain restarts the top-most one."""
    canvas = make_napari_viewer().window._qt_viewer.canvas.native
    first = NapariQtNotification.from_notification(
        Notification('first'), canvas
    )
    second = NapariQtNotification.from_notification(
        Notification('second'), canvas
    )
    first.show()
    second.show()

    starts, stops = [], []
    monkeypatch.setattr(
        NapariQtNotification, 'timer_start', lambda self: starts.append(self)
    )
    monkeypatch.setattr(
        NapariQtNotification, 'timer_stop', lambda self: stops.append(self)
    )

    qapp.focusChanged.emit(None, canvas)  # window gains focus
    assert starts == [second]

    qapp.focusChanged.emit(canvas, None)  # window loses focus
    assert stops == [first, second]

    first.close()
    second.close()


def test_focus_gain_does_not_restart_timer_of_hovered_notification(
    make_napari_viewer, monkeypatch, qapp
):
    """The timer of a hovered notification is not restarted on focus gain,
    as the user is reading it."""
    canvas = make_napari_viewer().window._qt_viewer.canvas.native
    notif = NapariQtNotification.from_notification(Notification('hi'), canvas)
    notif.show()
    notif.setAttribute(Qt.WidgetAttribute.WA_UnderMouse, True)

    starts = []
    monkeypatch.setattr(
        NapariQtNotification, 'timer_start', lambda self: starts.append(self)
    )

    qapp.focusChanged.emit(None, canvas)

    assert starts == []
    notif.close()
