"""Widgets for QtRender.

These were created for QtRender, but they are generic. Maybe we could switch
from these to something more napari-standard. These were just created
naively without knowing too much about napari's Qt code.
"""
from typing import Callable

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QWidget,
)


class QtLabeledSpinBox(QWidget):
    """A label plus a SpinBox.

    Parameters
    ----------
    label_text : str
        The label shown next to the spin box.
    initial_value : int
        The initial value of the spin box.
    spin_range : range
        The min/max/step of the spin box.
    connect : Callable[[int], None]
        Called when the user changes the value of the spin box.

    Attributes
    ----------
    spin : QSpinBox
        The spin box.
    """

    def __init__(
        self,
        label_text: str,
        initial_value: int,
        spin_range: range,
        connect: Callable[[int], None] = None,
    ):
        super().__init__()

        self._connect = connect
        layout = QHBoxLayout()

        self.spin = self._create(initial_value, spin_range)

        layout.addWidget(QLabel(label_text))
        layout.addWidget(self.spin)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

    def _create(self, initial_value: int, spin_range: range) -> QSpinBox:
        """Return the configured QSpinBox.

        Parameters
        ----------
        initial_value : int
            The initial value of the QSpinBox.
        spin_range : range
            The start/stop/step of the QSpinBox.

        Return
        ------
        QSpinBox
            The configured QSpinBox.
        """
        spin = QSpinBox()

        spin.setMinimum(spin_range.start)
        spin.setMaximum(spin_range.stop)
        spin.setSingleStep(spin_range.step)
        spin.setValue(initial_value)

        spin.setKeyboardTracking(False)  # correct?
        spin.setAlignment(Qt.AlignCenter)

        spin.valueChanged.connect(self._on_change)
        return spin

    def _on_change(self, value: int) -> None:
        """Called when the spin spin value was changed.

        value : int
            The new value of the SpinBox.
        """
        # We must clearFocus or it double-steps, no idea why.
        self.spin.clearFocus()

        if self._connect is not None:
            self._connect(value)  # Notify if we have a connection.


class QtLabeledComboBox(QWidget):
    """A generic ComboBox with a label.

    Provide an options dict. The keys will be displayed as the text options
    available to the user. The values are used by our set_value() and
    get_value() methods.

    Parameters
    ----------
    label : str
        The text label for the control.
    options : dict
        We display the keys and return the values.
    callback : Callable[[int], None]
        Called when the value is changed by the user.
    """

    def __init__(
        self, label: str, options: dict, callback: Callable[[int], None] = None
    ):
        super().__init__()
        self.options = options
        layout = QHBoxLayout()

        self.combo = QComboBox()
        self.combo.addItems(list(options.keys()))

        if callback is not None:
            self.combo.activated[int].connect(callback)

        layout.addWidget(QLabel(label))
        layout.addWidget(self.combo)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

    def set_value(self, value: str) -> None:
        """Make this value the selected item in the combo box.

        Parameters
        ----------
        value : str
            Set the combo box to this value.
        """
        for index, opt_value in enumerate(self.options.values()):
            if opt_value == value:
                self.combo.setCurrentIndex(index)

    def set_index(self, index: int) -> None:
        """Set what the combo box is showing based on the index.

        Parameters
        ----------
        index : int
            Set the combo box to this index.
        """
        self.combo.setCurrentIndex(index)

    def get_value(self) -> str:
        """Get the current value of the combo box.

        Return
        ------
        str
            The value currently selected in the combo box.
        """
        text = self.combo.currentText()
        return self.options[text]


class QtSimpleTable(QTableWidget):
    """A table of keys and values."""

    def __init__(self):
        super().__init__()
        self.verticalHeader().setVisible(False)
        self.horizontalHeader().setVisible(False)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.resizeRowsToContents()
        self.setShowGrid(False)

    def set_values(self, values: dict) -> None:
        """Populate the table with keys and values.

        values : dict
            Populate with these keys and values.
        """
        self.setRowCount(len(values))
        self.setColumnCount(2)
        for i, (key, value) in enumerate(values.items()):
            self.setItem(i, 0, QTableWidgetItem(key))
            self.setItem(i, 1, QTableWidgetItem(value))
