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
