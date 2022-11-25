from __future__ import annotations

from qtpy.QtWidgets import QLabel, QToolTip


class QtToolTipLabel(QLabel):
    """A QLabel that provides instant tooltips on mouser hover."""

    def enterEvent(self, event):
        """Override to show tooltips instantly."""
        if self.toolTip():
            pos = self.mapToGlobal(self.contentsRect().center())
            QToolTip.showText(pos, self.toolTip(), self)

        super().enterEvent(event)


class QtTopToolTipLabel(QLabel):
    """
    A QLabel that provides instant tooltips on mouser hover positioned over the label top.
    """

    def enterEvent(self, event):
        """
        Override to show tooltips instantly and set their position over the label top.
        """
        if self.toolTip():
            point = self.contentsRect().center()
            y_offset = self.contentsRect().height() * 3
            new_y = int(point.y() - y_offset)
            point.setY(new_y)
            pos = self.mapToGlobal(point)
            QToolTip.showText(pos, self.toolTip(), self)

        super().enterEvent(event)
