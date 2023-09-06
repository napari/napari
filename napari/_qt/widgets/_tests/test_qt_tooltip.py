import os
import sys

import pytest
from qtpy.QtCore import QPointF
from qtpy.QtGui import QEnterEvent
from qtpy.QtWidgets import QToolTip

from napari._qt.widgets.qt_tooltip import QtToolTipLabel


@pytest.mark.skipif(
    sys.platform == "darwin" and os.environ.get("CI", False),
    reason="Timeouts when running on macOS CI",
)
def test_qt_tooltip_label(qtbot):
    tooltip_text = "Test QtToolTipLabel showing a tooltip"
    widget = QtToolTipLabel("Label with a tooltip")
    widget.setToolTip(tooltip_text)
    qtbot.addWidget(widget)
    widget.show()

    assert QToolTip.text() == ""
    # put mouse outside of widget
    pos = QPointF(widget.rect().center())
    event = QEnterEvent(pos, pos, QPointF(widget.pos()) + pos)
    widget.enterEvent(event)
    qtbot.waitUntil(lambda: QToolTip.isVisible())
    qtbot.waitUntil(lambda: QToolTip.text() == tooltip_text)
