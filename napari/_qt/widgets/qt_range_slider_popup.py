from functools import partial

from qtpy.QtCore import QPoint, Qt
from qtpy.QtGui import QDoubleValidator, QFont, QFontMetrics
from qtpy.QtWidgets import QHBoxLayout, QLineEdit

from ...utils.translations import trans
from ..dialogs.qt_modal import QtPopup
from ..utils import qt_signals_blocked
from .qt_range_slider import QHRangeSlider, QVRangeSlider


class LabelEdit(QLineEdit):
    """Helper class to position LineEdits above the slider handle

    Parameters
    ----------
    value : str, optional
        Starting value, by default ''
    parent : QRangeSliderPopup, optional
        Required for proper label positioning above handle, by default None.
    get_pos : callable, optional
        Function that returns the position of the appropriate slider handle
        by default None.

    Attributes
    ----------
    get_pos : callable or None
        Function that returns the position of the appropriate slider handle.
    max_width : int
        Minimum label width.
    min_width : int
        Maximum label width.
    slider : qtpy.QtWidgets.QHRangeSlider
        Slider widget.
    """

    def __init__(self, value='', parent=None, get_pos=None):
        super().__init__(value, parent=parent)
        self.fm = QFontMetrics(QFont("", 0))
        self.setObjectName('slice_label')
        self.min_width = 30
        self.max_width = 200
        self.setCursor(Qt.IBeamCursor)
        self.setValidator(QDoubleValidator())
        self.textChanged.connect(self._on_text_changed)
        self._on_text_changed(value)

        self.get_pos = get_pos
        if parent is not None:
            self.min_width = 50
            self.slider = parent.slider
            self.setAlignment(Qt.AlignCenter)

    def _on_text_changed(self, text):
        """Update label text displayed above the slider handle.

        Parameters
        ----------
        text : str
            Label text to display above the slider handle.
        """
        # with non mono-spaced fonts, an "n-digit" number isn't always the same
        # width... so we convert all numbers to "n 8s" before measuring width
        # so as to avoid visual jitter in the width of the label
        width = self.fm.boundingRect('8' * len(text)).width() + 4
        width = max(self.min_width, min(width, self.max_width))
        if width > self.min_width:
            # don't ever make the label smaller ... it causes visual jitter
            self.min_width = width
        self.setFixedWidth(width)

    def update_position(self):
        """Update slider position."""
        x = self.get_pos() - self.width() // 2
        y = self.slider.handle_radius + 12
        self.move(QPoint(x, -y) + self.slider.pos())

    def mouseDoubleClickEvent(self, event):
        """Select all on mouse double click.

        Parameters
        ----------
        event : qtpy.QtCore.QEvent
            Event from the Qt context.
        """
        self.selectAll()


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
        self.slider = (
            QHRangeSlider(parent=parent, **kwargs)
            if horizontal
            else QVRangeSlider(parent=parent, **kwargs)
        )
        self.slider.setMinimumHeight(18)
        self.slider.setFocus()
        self.slider.valuesChanged.connect(self._on_values_change)
        self.slider.rangeChanged.connect(self._on_range_change)
        self.slider.resized.connect(self._update_cur_label_positions)

        # create "floating" min/max value labels
        cmin, cmax = self.slider.values()
        get_min_pos = partial(getattr, self.slider, 'display_min')
        get_max_pos = partial(getattr, self.slider, 'display_max')
        self.curmin_label = LabelEdit(self._numformat(cmin), self, get_min_pos)
        self.curmax_label = LabelEdit(self._numformat(cmax), self, get_max_pos)
        self.curmin_label.editingFinished.connect(self._curmin_label_changed)
        self.curmax_label.editingFinished.connect(self._curmax_label_changed)
        self.curmin_label.setToolTip(trans._("current minimum contrast limit"))
        self.curmax_label.setToolTip(trans._("current maximum contrast limit"))

        # create range min/max labels (left & right of slider)
        rmin, rmax = self.slider.range()
        self.range_min_label = LabelEdit(self._numformat(rmin))
        self.range_max_label = LabelEdit(self._numformat(rmax))
        self.range_min_label.editingFinished.connect(self._range_label_changed)
        self.range_max_label.editingFinished.connect(self._range_label_changed)
        self.range_min_label.setToolTip(trans._("minimum contrast range"))
        self.range_max_label.setToolTip(trans._("maximum contrast range"))
        self.range_min_label.setAlignment(Qt.AlignRight)

        # add widgets to layout
        self.layout = QHBoxLayout()
        self.frame.setLayout(self.layout)
        self.frame.setContentsMargins(0, 8, 0, 0)
        self.layout.addWidget(self.range_min_label)
        self.layout.addWidget(self.slider, 50)
        self.layout.addWidget(self.range_max_label)

    def _numformat(self, number):
        """Format float value to a specific number of decimal places.

        Parameters
        ----------
        number : float
            Number value formatted to a specific number of decimal places.
        """
        if round(number) == number:
            return "{:.{}f}".format(number, 0)
        else:
            return "{:.{}f}".format(number, self.precision)

    def _update_cur_label_positions(self):
        """Update label positions of current minimum/maximum contrast range."""
        self.curmin_label.update_position()
        self.curmax_label.update_position()

    def _on_values_change(self, values):
        """Update labels of the current contrast range.

        Parameters
        ----------
        values : tuple(float, float)
            Minimum and maximum values of the current contrast range.
        """
        cmin_, cmax_ = values
        with qt_signals_blocked(self.slider):
            self.curmin_label.setText(self._numformat(cmin_))
            self.curmax_label.setText(self._numformat(cmax_))
            self._update_cur_label_positions()

    def _on_range_change(self, values):
        """Update values of current contrast range and display labels.

        Parameters
        ----------
        values : tuple(float, float)
            Minimum and maximum values of the current contrast range.
        """
        cmin_, cmax_ = values
        with qt_signals_blocked(self.slider):
            self.range_min_label.setText(self._numformat(cmin_))
            self.range_max_label.setText(self._numformat(cmax_))
            # changing range may also change values
            vmin_, vmax_ = self.slider.values()
            self.curmin_label.setText(self._numformat(vmin_))
            self.curmax_label.setText(self._numformat(vmax_))

    def _curmin_label_changed(self):
        """Update minimum value of current contrast range."""
        cmin = float(self.curmin_label.text())
        cmax = float(self.curmax_label.text())
        if cmin > cmax:
            cmin = cmax
        self.slider.setValues((cmin, cmax))

    def _curmax_label_changed(self):
        """Update maximum value of current contrast range."""
        cmin = float(self.curmin_label.text())
        cmax = float(self.curmax_label.text())
        if cmax < cmin:
            cmax = cmin
        self.slider.setValues((cmin, cmax))

    def _range_label_changed(self):
        """Update values for minimum & maximum slider range to match labels."""
        rmin = float(self.range_min_label.text())
        rmax = float(self.range_max_label.text())
        if rmin >= rmax:
            rmax = rmin + 1
        self.slider.setRange((rmin, rmax))

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
