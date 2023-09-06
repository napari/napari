from qtpy.QtCore import QPoint
from qtpy.QtWidgets import QToolTip

from napari._qt.widgets.qt_tooltip import QtToolTipLabel


def test_qt_tooltip_label(qtbot):
    tooltip_text = "Test QtToolTipLabel showing a tooltip"
    widget = QtToolTipLabel("Label with a tooltip")
    widget.setToolTip(tooltip_text)
    qtbot.addWidget(widget)
    widget.show()

    assert QToolTip.text() == ""
    # put mouse outside of widget
    qtbot.mouseMove(widget, widget.rect().bottomRight() + QPoint(10, 10))
    qtbot.wait(50)
    qtbot.mouseMove(widget)

    qtbot.waitUntil(lambda: QToolTip.isVisible())
    qtbot.waitUntil(lambda: QToolTip.text() == tooltip_text)
