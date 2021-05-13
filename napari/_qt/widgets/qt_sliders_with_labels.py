from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QFont
from qtpy.QtWidgets import QHBoxLayout, QLabel, QLineEdit, QSlider, QWidget

from ...utils.translations import translator
from ..widgets.qt_range_slider import QHRangeSlider

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

        # self.setGeometry(0, 0, 30, 100)
        self._value = value
        self._min_value = min_value
        self._max_value = max_value
        self._single_step = single_step

        # Widget
        self._lineedit = QLineEdit()
        self._slider = QSlider(Qt.Horizontal, parent=parent)
        self._slider_min_label = QLabel(self)
        self._slider_max_label = QLabel(self)
        # self._validator = QIntValidator(min_value, max_value, self)

        # Widgets setup
        # self._lineedit.setValidator(self._validator)
        self._lineedit.setAlignment(Qt.AlignRight)
        self._lineedit.setAlignment(Qt.AlignHCenter)
        self._lineedit.setFixedWidth(25)
        self._slider_min_label.setText(str(min_value))
        self._slider_max_label.setText(str(max_value))
        self._slider.setMinimum(min_value)
        self._slider.setMaximum(max_value)

        font10 = QFont()
        font10.setPointSize(10)
        self._slider_min_label.setFont(font10)
        self._slider_max_label.setFont(font10)
        self._lineedit.setFont(font10)
        # self._slider.setFixedWidth(60)

        # Signals
        self._slider.valueChanged.connect(self._update_value)
        self._lineedit.textChanged.connect(self._update_value)

        # layout1 = QHBoxLayout()
        # layout1.addWidget(self._slider_min_label)
        # layout1.setContentsMargins(5,0,0,0)
        # layout1.setSpacing(0)
        # layout2 = QHBoxLayout()
        # layout2.addWidget(self._slider_max_label)
        # layout2.setContentsMargins(50,0, 0,0)
        # layout2.setSpacing(0)

        # layout3 = QHBoxLayout()
        # layout3.addLayout(layout1)
        # layout3.addLayout(layout2)

        # layout4 = QHBoxLayout()
        # layout4.addWidget(self._slider)
        # layout4.setContentsMargins(0,0,0,0)
        # layout4.setSpacing(0)

        # layout = QVBoxLayout()
        # layout.addLayout(layout4, 5)
        # layout.addLayout(layout3)

        # layout
        layout = QHBoxLayout()
        layout.addWidget(self._lineedit)
        layout.addWidget(self._slider_min_label)
        layout.addWidget(self._slider)
        layout.addWidget(self._slider_max_label)
        # layout.setAlignment(Qt.AlignBottom)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setAlignment(Qt.AlignHCenter)

        self.setLayout(layout)

        self._refresh()

    def setFocusPolicy(self, policy):
        """Set slider focus policy

        Parameters
        ----------
        policy: Qt.focusPolicy (check this)
        """
        self._slider.setFocusPolicy(policy)

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


class QtLabeledSlider2(QWidget):
    """Creates custom slider widget with 2 input text box.

    Parameters
    ----------

    """

    valuesChanged = Signal(tuple)
    rangeChanged = Signal(tuple)
    # self.layer.contrast_limits,
    #         self.layer.contrast_limits_range,
    #         parent=self,

    def __init__(
        self,
        initial_values=None,
        data_range=None,
        step_size=None,
        collapsible=True,
        collapsed=False,
        parent=None,
    ):
        super().__init__(parent)

        # self.setGeometry(0, 0, 30, 100)
        self._initial_values = initial_values
        self._data_range = data_range
        self._step_size = step_size
        # self._value = value
        # self._min_value = min_value
        # self._max_value = max_value
        # self._single_step = single_step

        # Widget
        self._lineedit1 = QLineEdit()
        self._lineedit2 = QLineEdit()
        self._slider = QHRangeSlider(initial_values, data_range, parent=parent)
        self._slider_min_label = QLabel(self)
        self._slider_max_label = QLabel(self)
        # self._validator = QIntValidator(min_value, max_value, self)

        # Widgets setup
        # self._lineedit.setValidator(self._validator)
        self._lineedit1.setAlignment(Qt.AlignRight)
        self._lineedit1.setAlignment(Qt.AlignHCenter)
        self._lineedit1.setFixedWidth(25)
        self._lineedit2.setAlignment(Qt.AlignRight)
        self._lineedit2.setAlignment(Qt.AlignHCenter)
        self._lineedit2.setFixedWidth(25)
        self._slider_min_label.setText(str(data_range[0]))
        self._slider_max_label.setText(str(data_range[1]))

        font10 = QFont()
        font10.setPointSize(10)
        self._slider_min_label.setFont(font10)
        self._slider_max_label.setFont(font10)
        self._lineedit1.setFont(font10)
        self._lineedit2.setFont(font10)
        # self._slider.setFixedWidth(60)

        # Signals
        self._slider.valuesChanged.connect(self._update_value)
        self._lineedit.textChanged.connect(self._update_value)

        # layout
        layout = QHBoxLayout()
        layout.addWidget(self._lineedit)
        layout.addWidget(self._slider_min_label)
        layout.addWidget(self._slider)
        layout.addWidget(self._slider_max_label)
        # layout.setAlignment(Qt.AlignBottom)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setAlignment(Qt.AlignHCenter)

        self.setLayout(layout)

        self._refresh()

    # def _update_value(self, value):
    #     """Update slider widget value.

    #     Parameters
    #     ----------
    #     value : int
    #         Widget value.
    #     """
    #     if value == "":
    #         value = int(self._value)

    #     value = int(value)

    #     if value > self._max_value:
    #         value = self._max_value
    #     elif value < self._min_value:
    #         value = self._min_value

    #     if value != self._value:
    #         self.valueChanged.emit(value)

    #     self._value = value
    #     self._refresh()

    # def _refresh(self):
    #     """Set every widget value to the new set value."""
    #     self.blockSignals(True)
    #     self._lineedit.setText(str(self._value))
    #     self._slider.setValue(self._value)
    #     self.blockSignals(False)
    #     # self.valueChanged.emit(self._value)

    # def setSingleStep(self, value):
    #     """sets the single step of the slider"""

    #     self._slider.setSingleStep(value)

    # def value(self):
    #     """Return current value.

    #     Returns
    #     -------
    #     int
    #         Current value of highlight widget.
    #     """
    #     return self._value

    # def setValue(self, value):
    #     """Set new value and update widget.

    #     Parameters
    #     ----------
    #     value : int
    #         Highlight value.
    #     """
    #     self._update_value(value)
    #     # self._refresh()

    # def setMinimum(self, value):
    #     """Set minimum widget value for slider.

    #     Parameters
    #     ----------
    #     value : int
    #         Minimum widget value.
    #     """
    #     value = int(value)
    #     if value < self._max_value:
    #         self._min_value = value
    #         self._slider_min_label.setText(str(value))
    #         self._slider.setMinimum(value)
    #         self._value = (
    #             self._min_value
    #             if self._value < self._min_value
    #             else self._value
    #         )
    #         self._refresh()
    #     else:
    #         raise ValueError(
    #             trans._(
    #                 "Minimum value must be smaller than {max_value}",
    #                 deferred=True,
    #                 max_value=self._max_value,
    #             )
    #         )

    # def minimum(self):
    #     """Return minimum widget value.

    #     Returns
    #     -------
    #     int
    #         Minimum value of widget widget.
    #     """
    #     return self._min_value

    # def setMaximum(self, value):
    #     """Set maximum widget value.

    #     Parameters
    #     ----------
    #     value : int
    #         Maximum widget value.
    #     """
    #     value = int(value)
    #     if value > self._min_value:
    #         self._max_value = value
    #         self._slider_max_label.setText(str(value))
    #         self._slider.setMaximum(value)
    #         self._value = (
    #             self._max_value
    #             if self._value > self._max_value
    #             else self._value
    #         )
    #         self._refresh()
    #     else:
    #         raise ValueError(
    #             trans._(
    #                 "Maximum value must be larger than {min_value}",
    #                 deferred=True,
    #                 min_value=self._min_value,
    #             )
    #         )

    # def maximum(self):
    #     """Return maximum widget value.

    #     Returns
    #     -------
    #     int
    #         Maximum value of highlight widget.
    #     """
    #     return self._max_value
