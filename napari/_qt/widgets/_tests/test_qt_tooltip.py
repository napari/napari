import sys

import pytest
from qtpy.QtWidgets import QToolTip

from napari._qt.widgets.qt_tooltip import QtToolTipLabel


@pytest.mark.skipif(
    sys.platform.startswith('linux') or sys.platform == 'darwin',
    reason='Timeouts when running on CI with Linux or macOS',
)
def test_qt_tooltip_label(qtbot):
    tooltip_text = "Test QtToolTipLabel showing a tooltip"
    widget = QtToolTipLabel("Label with a tooltip")
    widget.setToolTip(tooltip_text)
    qtbot.addWidget(widget)
    widget.show()

    assert QToolTip.text() == ""
    qtbot.mouseMove(widget)
    qtbot.waitUntil(lambda: QToolTip.isVisible())
    qtbot.waitUntil(lambda: QToolTip.text() == tooltip_text)
