from qtpy.QtCore import Qt
from qtpy.QtWidgets import QHBoxLayout
from qtrangeslider import QLabeledRangeSlider

from ..dialogs.qt_modal import QtPopup
from ..utils import ResizeFilter


class QRangeSliderPopup(QtPopup):
    def __init__(self, parent=None, horizontal=True, precision=0, **kwargs):
        """A popup window that contains a range slider and linked LineEdits.

        Parameters
        ----------
        parent : QWidget, optional
            Will like be an instance of QtLayerControls.  Note, providing
            parent can be useful to inherit stylesheets.
        horizontal : bool, optional
            Whether the slider is oriented horizontally, by default True.
            (Vertical orientation has not been tested much)
        precision : int, optional
            Number of decimal values in the labels, by default 0
        **kwargs
            all additional keyword arguments will be passed to the RangeSlider

        Attributes
        ----------
        curmax_label : napari._qt.qt_range_slider_popup.LabelEdit
            Label for the current maximum contrast limit.
        curmin_label : napari._qt.qt_range_slider_popup.LabelEdit
            Label for the current minimum contrast limit.
        frame : qtpy.QtWidgets.QFrame
            Frame of the popup dialog box.
        layout : qtpy.QtWidgets.QHBoxLayout
            Layout of the popup dialog box.
        precision : int
            Number of decimal values in numeric labels.
        range_max_label : napari._qt.qt_range_slider_popup.LabelEdit
            Label for maximum slider range value.
        range_min_label : napari._qt.qt_range_slider_popup.LabelEdit
            Label for minimum slider range value.
        slider : qtpy.QtWidgets.QHRangeSlider
            Slider widget.
        """
        super().__init__(parent)
        self.precision = precision

        # create slider
        orient = Qt.Horizontal if horizontal else Qt.Vertical
        self.slider = QLabeledRangeSlider(orient, self)
        self.slider.setFocus()

        resize_filter = ResizeFilter()
        self.slider.installEventFilter(resize_filter)

        # add widgets to layout
        self._layout = QHBoxLayout()
        self.frame.setLayout(self._layout)
        # self.frame.setContentsMargins(0, 8, 0, 0)
        self._layout.addWidget(self.slider)

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

    def closeEvent(self, event):
        self.deleteLater()
        event.accept()
