import os
import sys
from unittest.mock import patch

import pytest
from qtpy.QtCore import QPointF
from qtpy.QtGui import QEnterEvent
from qtpy.QtWidgets import QToolTip

from napari._qt.widgets.qt_tooltip import QtToolTipLabel


@pytest.mark.skipif(
    os.environ.get('CI', False) and sys.platform == 'darwin',
    reason='Timeouts when running on macOS CI',
)
@patch.object(QToolTip, 'showText')
def test_qt_tooltip_label(show_text, qtbot):
    tooltip_text = 'Test QtToolTipLabel showing a tooltip'
    widget = QtToolTipLabel('Label with a tooltip')
    widget.setToolTip(tooltip_text)
    qtbot.addWidget(widget)
    widget.show()

    assert QToolTip.text() == ''
    # simulate movement mouse from outside the widget to the center
    pos = QPointF(widget.rect().center())
    event = QEnterEvent(pos, pos, QPointF(widget.pos()) + pos)
    widget.enterEvent(event)
    assert show_text.called
    assert show_text.call_args[0][1] == tooltip_text
