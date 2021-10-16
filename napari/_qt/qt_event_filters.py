"""Qt event filters providing custom handling of events."""

import html

from superqt.qtcompat.QtCore import QEvent, QObject
from superqt.qtcompat.QtWidgets import QWidget

from .utils import qt_might_be_rich_text


class QtToolTipEventFilter(QObject):
    """
    An event filter that converts all plain-text widget tooltips to rich-text
    tooltips.
    """

    def eventFilter(self, qobject: QObject, event: QEvent) -> bool:
        if event.type() == QEvent.Type.ToolTipChange and isinstance(
            qobject, QWidget
        ):
            tooltip = qobject.toolTip()
            if tooltip and not qt_might_be_rich_text(tooltip):
                qobject.setToolTip(f'<qt>{html.escape(tooltip)}</qt>')
                return True

        return super().eventFilter(qobject, event)
