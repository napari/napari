from __future__ import annotations

from typing import TYPE_CHECKING

from qtpy.QtCore import Qt
from qtpy.QtWidgets import QApplication, QHBoxLayout
from superqt import QLabeledDoubleRangeSlider

from napari._qt.dialogs.qt_modal import QtPopup

if TYPE_CHECKING:
    from qtpy.QtGui import QKeyEvent
    from qtpy.QtWidgets import QWidget


class QRangeSliderPopup(QtPopup):
    """A popup window that contains a labeled range slider and buttons.

    Parameters
    ----------
    parent : QWidget, optional
        Will like be an instance of QtLayerControls.  Note, providing
        parent can be useful to inherit stylesheets.

    Attributes
    ----------
    slider : QLabeledRangeSlider
        Slider widget.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        # create slider
        self.slider = QLabeledDoubleRangeSlider(
            Qt.Orientation.Horizontal, parent
        )
        self.slider.label_shift_x = 2
        self.slider.label_shift_y = 2
        self.slider.setFocus()

        # add widgets to layout
        self._layout = QHBoxLayout()
        self._layout.setContentsMargins(10, 0, 10, 16)
        self.frame.setLayout(self._layout)
        self._layout.addWidget(self.slider)
        QApplication.processEvents()
        self.slider._reposition_labels()

    def keyPressEvent(self, event: QKeyEvent | None) -> None:
        """On key press lose focus of the lineEdits.

        Parameters
        ----------
        event : qtpy.QtCore.QKeyEvent
            Event from the Qt context.
        """
        # we override the parent keyPressEvent so that hitting enter does not
        # hide the window... but we do want to lose focus on the lineEdits
        if event is not None and event.key() in (
            Qt.Key.Key_Return,
            Qt.Key.Key_Enter,
        ):
            self.slider.setFocus()
            return
        super().keyPressEvent(event)  # type: ignore[no-untyped-call]
