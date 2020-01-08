from functools import partial

from qtpy.QtCore import Qt, QPoint
from qtpy.QtGui import QDoubleValidator, QFontMetrics, QFont
from qtpy.QtWidgets import QHBoxLayout, QLineEdit

from .qt_modal import QtPopup
from .qt_range_slider import QHRangeSlider, QVRangeSlider
from .utils import qt_signals_blocked


class LabelEdit(QLineEdit):
    def __init__(self, value='', parent=None, get_pos=None):
        """Helper class to position LineEdits above the slider handle

        Parameters
        ----------
        value : str, optional
            starting value, by default ''
        parent : QRangeSliderPopup, optional
            required for proper label positioning above handle, by default None
        get_pos : callable, optional
            function that returns the position of the appropriate slider handle
            by default None
        """
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
        x = self.get_pos() - self.width() / 2
        y = self.slider.handle_radius + 6
        self.move(QPoint(x, -y) + self.slider.pos())

    def mouseDoubleClickEvent(self, event):
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
        self.curmin_label.setToolTip("current minimum contrast limit")
        self.curmax_label.setToolTip("current maximum contrast limit")

        # create range min/max labels (left & right of slider)
        rmin, rmax = self.slider.range()
        self.range_min_label = LabelEdit(self._numformat(rmin))
        self.range_max_label = LabelEdit(self._numformat(rmax))
        self.range_min_label.editingFinished.connect(self._range_label_changed)
        self.range_max_label.editingFinished.connect(self._range_label_changed)
        self.range_min_label.setToolTip("minimum contrast range")
        self.range_max_label.setToolTip("maximum contrast range")
        self.range_min_label.setAlignment(Qt.AlignRight)

        # add widgets to layout
        self.layout = QHBoxLayout()
        self.frame.setLayout(self.layout)
        self.frame.setContentsMargins(0, 8, 0, 0)
        self.layout.addWidget(self.range_min_label)
        self.layout.addWidget(self.slider, 50)
        self.layout.addWidget(self.range_max_label)

    def _numformat(self, number):
        if round(number) == number:
            return "{:.{}f}".format(number, 0)
        else:
            return "{:.{}f}".format(number, self.precision)

    def _update_cur_label_positions(self):
        self.curmin_label.update_position()
        self.curmax_label.update_position()

    def _on_values_change(self, values):
        cmin_, cmax_ = values
        with qt_signals_blocked(self.slider):
            self.curmin_label.setText(self._numformat(cmin_))
            self.curmax_label.setText(self._numformat(cmax_))
            self._update_cur_label_positions()

    def _on_range_change(self, values):
        cmin_, cmax_ = values
        with qt_signals_blocked(self.slider):
            self.range_min_label.setText(self._numformat(cmin_))
            self.range_max_label.setText(self._numformat(cmax_))
            # changing range may also change values
            vmin_, vmax_ = self.slider.values()
            self.curmin_label.setText(self._numformat(vmin_))
            self.curmax_label.setText(self._numformat(vmax_))

    def _curmin_label_changed(self):
        cmin = float(self.curmin_label.text())
        cmax = float(self.curmax_label.text())
        if cmin > cmax:
            cmin = cmax
        self.slider.setValues((cmin, cmax))

    def _curmax_label_changed(self):
        cmin = float(self.curmin_label.text())
        cmax = float(self.curmax_label.text())
        if cmax < cmin:
            cmax = cmin
        self.slider.setValues((cmin, cmax))

    def _range_label_changed(self):
        rmin = float(self.range_min_label.text())
        rmax = float(self.range_max_label.text())
        if rmin >= rmax:
            rmax = rmin + 1
        self.slider.setRange((rmin, rmax))

    def keyPressEvent(self, event):
        # we override the parent keyPressEvent so that hitting enter does not
        # hide the window... but we do want to lose focus on the lineEdits
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            self.slider.setFocus()
            return
        super().keyPressEvent(event)
