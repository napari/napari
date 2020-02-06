from qtpy.QtCore import Qt, QPoint
from qtpy.QtGui import QDoubleValidator, QFontMetrics, QFont
from qtpy.QtWidgets import QHBoxLayout, QLineEdit, QSlider

from .qt_modal import QtPopup
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
        x = self.get_pos() * 4 - self.width() / 2
        y = 8 + 6
        self.move(QPoint(x, -y) + self.slider.pos())

    def mouseDoubleClickEvent(self, event):
        self.selectAll()


class QSliderPopup(QtPopup):
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
            QSlider(Qt.Horizontal) if horizontal else QSlider(Qt.Vertical)
        )
        self.slider.setMinimum(2)
        self.slider.setMaximum(200)
        self.slider.setSingleStep(1)
        self.slider.setValue(100)
        self.slider.setMinimumHeight(18)
        self.slider.setFocus()
        self.slider.valueChanged.connect(self._on_value_change)
        # self.slider.resized.connect(self._update_cur_label_positions) # no resized signal yet

        # create "floating" value label
        cval = self.slider.value()
        get_val_pos = self.slider.value  # TODO: validate
        self.cval_label = LabelEdit(self._numformat(cval), self, get_val_pos)
        self.cval_label.editingFinished.connect(self._curval_label_changed)
        self.cval_label.setToolTip("current gamma")

        # add widgets to layout
        self.layout = QHBoxLayout()
        self.frame.setLayout(self.layout)
        self.frame.setContentsMargins(0, 8, 0, 0)
        self.layout.addWidget(self.slider, 50)

    def _numformat(self, number):
        if round(number) == number:
            return "{:.{}f}".format(number, 0)
        else:
            return "{:.{}f}".format(number, self.precision)

    def _update_cur_label_positions(self):
        self.cval_label.update_position()

    def _on_value_change(self, value):
        with qt_signals_blocked(self.slider):
            self.cval_label.setText(str(float(value) / 100.0))
            self._update_cur_label_positions()

    def _curval_label_changed(self):
        cval = int(self.curmin_label.text())  # TODO: check int casting here
        self.slider.setValue(cval)

    def keyPressEvent(self, event):
        # we override the parent keyPressEvent so that hitting enter does not
        # hide the window... but we do want to lose focus on the lineEdits
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            self.slider.setFocus()
            return
        super().keyPressEvent(event)
