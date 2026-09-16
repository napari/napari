from __future__ import annotations

from typing import TYPE_CHECKING

from qtpy.QtWidgets import QLabel, QToolTip

if TYPE_CHECKING:
    from qtpy.QtGui import QEnterEvent


class QtToolTipLabel(QLabel):
    """A QLabel that provides instant tooltips on mouser hover."""

    def enterEvent(self, event: QEnterEvent | None) -> None:
        """Override to show tooltips instantly."""
        if self.toolTip():
            pos = self.mapToGlobal(self.contentsRect().center())
            QToolTip.showText(pos, self.toolTip(), self)

        super().enterEvent(event)
