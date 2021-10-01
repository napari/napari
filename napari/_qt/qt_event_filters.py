"""Qt event filters providing custom handling of some events."""

import html

from qtpy.QtCore import QEvent, QObject, Qt
from qtpy.QtWidgets import QWidget

from ..utils.translations import trans


class QtToolTipEventFilter(QObject):
    """
    An event filter that converts all plain-text widget tooltips to rich-text
    tooltips.
    """

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        if event.type() == QEvent.ToolTipChange and isinstance(obj, QWidget):
            tooltip = widget.toolTip()
            if tooltip and Qt.mightBeRichText(tooltip):
                widget.setToolTip(f'<qt>{html.escape(tooltip)}</qt>')
                return True

        return super().eventFilter(widget, event)
