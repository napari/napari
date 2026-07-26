import pytest

from napari._qt.widgets.qt_dims import QtDims, QtDimSliderWidget
from napari._qt.widgets.qt_dims_slider import SLIDER_MINIMUM_WIDTH
from napari.components import Dims


def test_same_margin_popup(qtbot):
    dims = Dims(ndim=3)
    view = QtDims(dims)
    qtbot.addWidget(view)
    slider: QtDimSliderWidget = view.slider_widgets[0]
    # Lazily create the widget
    assert slider.margins_popup is None
    slider.show_margins_popup()
    old_margins_popup = slider.margins_popup
    assert old_margins_popup is not None
    # Reuse old margins popup
    slider.show_margins_popup()
    assert old_margins_popup is slider.margins_popup


def test_move_margin_popup(qtbot):
    dims = Dims(ndim=3)
    view = QtDims(dims)
    qtbot.addWidget(view)
    slider: QtDimSliderWidget = view.slider_widgets[0]
    slider.show_margins_popup()
    # Check that values of the left slider matches the
    # values of the dims margin_right after the
    # margin_right has been moved within the dims
    dims.margin_right = (2, 0, 0)
    assert slider.margins_popup.right_slider.value() == dims.margin_right[0]
    slider.margins_popup.left_slider.setValue(1)
    assert slider.margins_popup.left_slider.value() == dims.margin_left[0]


def test_slider_has_a_minimum_width(qtbot):
    """The groove must not be squeezed to nothing by a narrow window.

    Without a minimum the row's minimum width comes only from its labels and
    the slider absorbs the whole shortfall, leaving a groove too short to
    position the handle on an axis with many steps.
    """
    dims = Dims(ndim=3)
    view = QtDims(dims)
    qtbot.addWidget(view)
    for slider_widget in view.slider_widgets:
        assert slider_widget.slider.minimumWidth() == SLIDER_MINIMUM_WIDTH


def test_slider_minimum_width_reaches_the_row(qtbot):
    """The constraint propagates, so a narrow window cannot collapse it."""
    dims = Dims(ndim=3)
    view = QtDims(dims)
    qtbot.addWidget(view)
    slider_widget: QtDimSliderWidget = view.slider_widgets[0]
    assert slider_widget.minimumSizeHint().width() >= SLIDER_MINIMUM_WIDTH


# -- Locked-axis flash reminder ------------------------------------------------


def test_disabled_child_click_reaches_parent(qtbot):
    """Pin the Qt guarantee the lock-flash relies on.

    A mouse press on a *disabled* child is not consumed: Qt propagates it up to
    the first enabled ancestor. This is why the row can detect a click on the
    (disabled) slider of a locked axis in its own ``mousePressEvent`` without
    re-enabling the slider. If a future Qt/binding upgrade changed this, the
    pointer path of the lock reminder would silently stop working — hence a
    dedicated guard rather than trusting it implicitly.
    """
    from qtpy.QtCore import Qt
    from qtpy.QtTest import QTest
    from qtpy.QtWidgets import QScrollBar, QVBoxLayout, QWidget

    presses = []

    class Parent(QWidget):
        def mousePressEvent(self, event):
            presses.append(True)

    parent = Parent()
    qtbot.addWidget(parent)
    layout = QVBoxLayout(parent)
    scrollbar = QScrollBar(Qt.Orientation.Horizontal)
    scrollbar.setFixedSize(200, 20)
    layout.addWidget(scrollbar)
    parent.resize(240, 60)
    parent.show()
    qtbot.waitExposed(parent)

    center = scrollbar.mapTo(parent, scrollbar.rect().center())
    no_mod = Qt.KeyboardModifier.NoModifier

    # Enabled: the scrollbar handles the press itself, parent sees nothing.
    scrollbar.setEnabled(True)
    presses.clear()
    QTest.mouseClick(parent.windowHandle(), Qt.LeftButton, no_mod, center)
    assert presses == []

    # Disabled: the press propagates to the enabled parent.
    scrollbar.setEnabled(False)
    presses.clear()
    QTest.mouseClick(parent.windowHandle(), Qt.LeftButton, no_mod, center)
    assert presses == [True]


# The flash uses a real single-shot QTimer. The ``_dangling_qtimers`` guard in
# conftest trips when qtbot deletes the widget (and its child timer) before the
# guard's post-test check, so these tests use the sanctioned ``disable_qtimer_start``
# marker to no-op the timer and drive the reset step explicitly. The wall-clock
# reset itself is trivial wiring; the marker keeps the suite deterministic.


@pytest.mark.disable_qtimer_start
def test_blocked_step_emits_rejection_and_flashes(qtbot):
    """A key/step/editor navigation on a locked axis flashes its padlock.

    These paths never touch the disabled slider, so they route through
    ``Dims.events.axis_lock_rejected`` rather than the row's ``mousePressEvent``.
    """
    dims = Dims(ndim=3)
    view = QtDims(dims)
    qtbot.addWidget(view)
    dims.lock_axis(0)
    widget: QtDimSliderWidget = view.slider_widgets[0]
    assert not widget.lock_button.property('flash')

    rejected = []
    dims.events.axis_lock_rejected.connect(
        lambda e: rejected.append(tuple(e.value))
    )

    dims.set_current_step(0, 2)  # blocked: axis 0 is locked

    assert rejected == [(0,)]
    assert widget.lock_button.property('flash') is True
    # An unlocked axis is unaffected and does not flash.
    assert view.slider_widgets[1].lock_button.property('flash') is False


@pytest.mark.disable_qtimer_start
def test_press_on_locked_slider_flashes(qtbot):
    """The row detects a pointer poke on its frozen slider and flashes.

    Complements ``test_disabled_child_click_reaches_parent`` (which pins that
    the press reaches the row): here the row's own handler turns that press into
    a flash. A press landing over the disabled slider of a locked axis flashes;
    the same press when the axis is movable does not.
    """
    from qtpy.QtCore import QEvent, QPointF, Qt
    from qtpy.QtGui import QMouseEvent
    from qtpy.QtWidgets import QApplication

    dims = Dims(ndim=3)
    view = QtDims(dims)
    qtbot.addWidget(view)
    view.resize(600, 200)
    view.show()
    QApplication.processEvents()  # realize the layout so geometries are valid

    widget: QtDimSliderWidget = view.slider_widgets[0]

    def press_slider() -> QMouseEvent:
        center = widget.slider.geometry().center()
        event = QMouseEvent(
            QEvent.Type.MouseButtonPress,
            QPointF(center),
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        widget.mousePressEvent(event)

    # Movable axis: a press over the (enabled) slider must not flash.
    press_slider()
    assert widget.lock_button.property('flash') is False

    dims.lock_axis(0)
    assert not widget.slider.isEnabled()
    press_slider()
    assert widget.lock_button.property('flash') is True


@pytest.mark.disable_qtimer_start
def test_lock_flash_reset_clears_tint(qtbot):
    """The flash is transient: the timeout handler clears the amber tint."""
    dims = Dims(ndim=3)
    view = QtDims(dims)
    qtbot.addWidget(view)
    dims.lock_axis(0)
    widget: QtDimSliderWidget = view.slider_widgets[0]

    # The timer is wired single-shot at the documented duration...
    assert widget._lock_flash_timer.isSingleShot()

    widget._flash_lock()
    assert widget.lock_button.property('flash') is True
    widget._end_lock_flash()  # ...and firing it (simulated) clears the tint.
    assert widget.lock_button.property('flash') is False
