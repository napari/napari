from PyQt5.QtWidgets import QApplication
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QHBoxLayout
from qtrangeslider import QLabeledRangeSlider

from ..dialogs.qt_modal import QtPopup


class QRangeSliderPopup(QtPopup):
    def __init__(self, parent=None, precision=0):
        """A popup window that contains a range slider and linked LineEdits.

        Parameters
        ----------
        parent : QWidget, optional
            Will like be an instance of QtLayerControls.  Note, providing
            parent can be useful to inherit stylesheets.
        precision : int, optional
            Number of decimal values in the labels, by default 0

        Attributes
        ----------
        precision : int
            Number of decimal values in numeric labels.
        slider : QLabeledRangeSlider
            Slider widget.
        """
        super().__init__(parent)
        self.precision = precision

        # create slider
        self.slider = QLabeledRangeSlider(Qt.Horizontal, parent)
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

    def keyPressEvent(self, event):
        """On key press lose focus of the lineEdits.

        Parameters
        ----------
        event : qtpy.QtCore.QEvent
            Event from the Qt context.
        """
        # we override the parent keyPressEvent so that hitting enter does not
        # hide the window... but we do want to lose focus on the lineEdits
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            self.slider.setFocus()
            return
        super().keyPressEvent(event)
