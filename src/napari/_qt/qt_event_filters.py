"""Qt event filters providing custom handling of events."""

import html

from qtpy.QtCore import QEvent, QObject
from qtpy.QtWidgets import QWidget

from napari._qt.utils import qt_might_be_rich_text


class QtToolTipEventFilter(QObject):
    """
    An event filter that converts all plain-text widget tooltips to rich-text
    tooltips.
    """

    def eventFilter(self, qobject: QObject, event: QEvent) -> bool:
        if event.type() == QEvent.ToolTipChange and isinstance(
            qobject, QWidget
        ):
            tooltip = qobject.toolTip()
            if tooltip and not qt_might_be_rich_text(tooltip):
                qobject.setToolTip(f'<qt>{html.escape(tooltip)}</qt>')
                return True

        return super().eventFilter(qobject, event)
