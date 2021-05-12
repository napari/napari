from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import QHBoxLayout, QLabel, QLineEdit, QSlider, QWidget

from ...utils.translations import translator

trans = translator.load()


# from qtpy.QtCore import QSize, Qt, Signal
# from qtpy.QtGui import QColor, QIntValidator, QPainter, QPainterPath, QPen
# from qtpy.QtWidgets import (
#     QDialog,
#     QFrame,
#     QHBoxLayout,
#     QLabel,
#     QLineEdit,
#     QSlider,
#     QVBoxLayout,
#     QWidget,
# )

# from ...utils.translations import translator

# trans = translator.load()


class QtLabeledSlider1(QWidget):
    """Creates custom slider widget with 1 input text box.

    Parameters
    ----------

    """

    valueChanged = Signal(int)

    def __init__(
        self,
        parent: QWidget = None,
        value: int = 1,
        min_value: int = 1,
        max_value: int = 10,
        single_step: int = 1,
    ):
        super().__init__(parent)

        # self.setGeometry(300, 300, 125, 110)
        self._value = value
        self._min_value = min_value
        self._max_value = max_value
        self._single_step = single_step

        # Widget
        self._lineedit = QLineEdit()
        self._unit = QLabel(self)
        self._slider = QSlider(Qt.Horizontal, parent=parent)
        self._slider_min_label = QLabel(self)
        self._slider_max_label = QLabel(self)
        # self._validator = QIntValidator(min_value, max_value, self)

        # Widgets setup
        # self._lineedit.setValidator(self._validator)
        self._lineedit.setAlignment(Qt.AlignRight)
        self._lineedit.setAlignment(Qt.AlignBottom)
        self._slider_min_label.setText(str(min_value))
        self._slider_min_label.setAlignment(Qt.AlignBottom)
        self._slider_max_label.setText(str(max_value))
        self._slider_max_label.setAlignment(Qt.AlignBottom)
        self._slider.setMinimum(min_value)
        self._slider.setMaximum(max_value)

        # Signals
        self._slider.valueChanged.connect(self._update_value)
        self._lineedit.textChanged.connect(self._update_value)

        # layout
        layout = QHBoxLayout()
        layout.addWidget(self._lineedit)
        layout.addWidget(self._slider_min_label)
        layout.addWidget(self._slider)
        layout.addWidget(self._slider_max_label)
        layout.setAlignment(Qt.AlignBottom)

        self.setLayout(layout)

        self._refresh()

    def _update_value(self, value):
        """Update slider widget value.

        Parameters
        ----------
        value : int
            Widget value.
        """
        if value == "":
            value = int(self._value)

        value = int(value)

        if value > self._max_value:
            value = self._max_value
        elif value < self._min_value:
            value = self._min_value

        if value != self._value:
            self.valueChanged.emit(value)

        self._value = value
        self._refresh()

    def _refresh(self):
        """Set every widget value to the new set value."""
        self.blockSignals(True)
        self._lineedit.setText(str(self._value))
        self._slider.setValue(self._value)
        self.blockSignals(False)
        # self.valueChanged.emit(self._value)

    def setSingleStep(self, value):
        """sets the single step of the slider"""

        self._slider.setSingleStep(value)

    def value(self):
        """Return current value.

        Returns
        -------
        int
            Current value of highlight widget.
        """
        return self._value

    def setValue(self, value):
        """Set new value and update widget.

        Parameters
        ----------
        value : int
            Highlight value.
        """
        self._update_value(value)
        # self._refresh()

    def setMinimum(self, value):
        """Set minimum widget value for slider.

        Parameters
        ----------
        value : int
            Minimum widget value.
        """
        value = int(value)
        if value < self._max_value:
            self._min_value = value
            self._slider_min_label.setText(str(value))
            self._slider.setMinimum(value)
            self._value = (
                self._min_value
                if self._value < self._min_value
                else self._value
            )
            self._refresh()
        else:
            raise ValueError(
                trans._(
                    "Minimum value must be smaller than {max_value}",
                    deferred=True,
                    max_value=self._max_value,
                )
            )

    def minimum(self):
        """Return minimum widget value.

        Returns
        -------
        int
            Minimum value of widget widget.
        """
        return self._min_value

    def setMaximum(self, value):
        """Set maximum widget value.

        Parameters
        ----------
        value : int
            Maximum widget value.
        """
        value = int(value)
        if value > self._min_value:
            self._max_value = value
            self._slider_max_label.setText(str(value))
            self._slider.setMaximum(value)
            self._value = (
                self._max_value
                if self._value > self._max_value
                else self._value
            )
            self._refresh()
        else:
            raise ValueError(
                trans._(
                    "Maximum value must be larger than {min_value}",
                    deferred=True,
                    min_value=self._min_value,
                )
            )

    def maximum(self):
        """Return maximum widget value.

        Returns
        -------
        int
            Maximum value of highlight widget.
        """
        return self._max_value
