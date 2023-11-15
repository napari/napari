from qtpy.QtCore import Signal
from qtpy.QtWidgets import QHBoxLayout, QPushButton, QWidget

from napari._qt.widgets.qt_spinbox import QtSpinBox
from napari.settings import get_settings
from napari.utils.theme import get_system_theme, get_theme
from napari.utils.translations import trans


class QtFontSizeWidget(QWidget):
    """
    Widget to change `font_size` and enable to reset is value to the current
    selected theme default `font_size` value.
    """

    valueChanged = Signal(int)

    def __init__(self, parent: QWidget = None) -> None:
        super().__init__(parent=parent)
        self._spinbox = QtSpinBox()
        self._reset_button = QPushButton(trans._("Reset font size"))

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._spinbox)
        layout.addWidget(self._reset_button)
        self.setLayout(layout)

        self._spinbox.valueChanged.connect(self.valueChanged)
        self._reset_button.clicked.connect(self._reset)

    def _reset(self) -> None:
        """
        Reset the widget value to the current selected theme font size value.
        """
        current_theme_name = get_settings().appearance.theme
        if current_theme_name == "system":
            # system isn't a theme, so get the name
            current_theme_name = get_system_theme()
        current_theme = get_theme(current_theme_name)
        self.setValue(int(current_theme.font_size[:-2]))

    def value(self) -> int:
        """
        Return the current widget value.

        Returns
        -------
        int
            The current value.
        """
        return self._spinbox.value()

    def setValue(self, value: int) -> None:
        """
        Set the current widget value.

        Parameters
        ----------
        value : int
            The current value.
        """
        self._spinbox.setValue(value)

    def setRange(self, min_value: int, max_value: int) -> None:
        """
        Value range that the spinbox widget will use.

        Parameters
        ----------
        min_value : int
            Minimum value the font_size could be set.
        max_value : int
            Maximum value the font_size could be set.
        """
        self._spinbox.setRange(min_value, max_value)
