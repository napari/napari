from qtpy import QtGui
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QHBoxLayout, QLineEdit, QFrame
from .qt_range_slider import QHRangeSlider, QVRangeSlider
from .qt_modal import QtPopup
from .utils import qt_signals_blocked


class LabelEdit(QLineEdit):
    # Just a helper class to minimize a lot of code duplication
    def __init__(self, value='', parent=None):
        super().__init__(value, parent)
        self.setObjectName('slice_label')
        self.setFixedWidth(40)
        self.setValidator(QtGui.QDoubleValidator(0, 999999, 1))
        self.setCursor(Qt.IBeamCursor)


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
        layout = QHBoxLayout()
        self.slider = (
            QHRangeSlider(**kwargs) if horizontal else QVRangeSlider(**kwargs)
        )
        self.slider.setMinimumHeight(18)
        self.frame.setLayout(layout)
        cmin, cmax = self.slider.values()
        self.curmin_label = LabelEdit(self._numformat(cmin))
        self.curmax_label = LabelEdit(self._numformat(cmax))
        rmin, rmax = self.slider.range
        self.range_min_label = LabelEdit(self._numformat(rmin))
        self.range_max_label = LabelEdit(self._numformat(rmax))
        self.range_min_label.setAlignment(Qt.AlignRight)
        self.curmax_label.setAlignment(Qt.AlignRight)
        sep1 = QFrame(self)
        sep2 = QFrame(self)
        sep1.setFixedSize(1, 14)
        sep2.setFixedSize(1, 14)
        sep1.setObjectName('slice_label_sep')
        sep2.setObjectName('slice_label_sep')
        layout.addWidget(self.range_min_label)
        layout.addWidget(sep1)
        layout.addWidget(self.curmin_label)
        layout.addWidget(self.slider, 50)
        layout.addWidget(self.curmax_label)
        layout.addWidget(sep2)
        layout.addWidget(self.range_max_label)
        self.layout = layout
        self.slider.setFocus()
        self.curmin_label.editingFinished.connect(self._curmin_label_changed)
        self.curmax_label.editingFinished.connect(self._curmax_label_changed)
        self.range_min_label.editingFinished.connect(self._range_label_changed)
        self.range_max_label.editingFinished.connect(self._range_label_changed)
        self.slider.valuesChanged.connect(self._on_values_change)
        self.slider.rangeChanged.connect(self._on_range_change)

    def _numformat(self, number):
        return "{:.{}f}".format(number, self.precision)

    def _on_values_change(self, values):
        cmin_, cmax_ = values
        with qt_signals_blocked(self.slider):
            self.curmin_label.setText(self._numformat(cmin_))
            self.curmax_label.setText(self._numformat(cmax_))

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
