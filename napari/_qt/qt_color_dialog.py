import re
from typing import Optional

import numpy as np
from qtpy.QtCore import QEvent, Qt, Signal, Slot
from qtpy.QtGui import QColor
from qtpy.QtWidgets import (
    QColorDialog,
    QCompleter,
    QFrame,
    QHBoxLayout,
    QLineEdit,
    QVBoxLayout,
    QWidget,
)

from ..layers.utils.color_transformations import ColorType
from ..utils.colormaps.standardize_color import (
    hex_to_name,
    rgb_to_hex,
    transform_color,
)
from .qt_modal import QtPopup
from vispy.color import get_color_dict

# matches any 3- or 4-tuple of int or float, with or without parens
# captures the numbers into groups.
# this is used to allow users to enter colors as e.g.: "(1, 0.7, 0)"
rgba_regex = re.compile(
    r"\(?([\d.]+),\s*([\d.]+),\s*([\d.]+),?\s*([\d.]+)?\)?"
)


class QColorSwatchEdit(QWidget):
    """A widget that combines a QColorSwatch with a QColorLineEdit.

    emits a color_changed event with a 1x4 numpy array when the current color
    changes.  Note, the "model" for the current color is the ``_color``
    attribute on the QColorSwatch.
    """

    color_changed = Signal(np.ndarray)

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        *,
        initial_color: Optional[ColorType] = None,
        tooltip: Optional[str] = None,
    ):
        """Create a new ColorSwatchEdit widget.

        Parameters
        ----------
        parent : QWidget, optional
            parent widget, by default None
        initial_color : ColorType, optional
            Starting color, by default None
        tooltip : str, optional
            Tooltip when hovering on the swatch,
            by default 'click to set color'
        """
        super().__init__(parent=parent)
        self.setObjectName('QColorSwatchEdit')

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        self.setLayout(layout)

        self.lineEdit = QColorLineEdit(self)
        self.lineEdit.editingFinished.connect(self._on_lineedit_edited)

        self.colorSwatch = QColorSwatch(self, tooltip=tooltip)
        self.colorSwatch.color_changed.connect(self._on_swatch_changed)
        self.setColor = self.colorSwatch.setColor
        if initial_color is not None:
            self.setColor(initial_color)

        layout.addWidget(self.colorSwatch)
        layout.addWidget(self.lineEdit)

    def color(self):
        """Return the current color."""
        return self.colorSwatch.color()

    def _on_lineedit_edited(self):
        """When the user hits enter or loses focus on the LineEdit widget."""
        text = self.lineEdit.text()
        rgb_match = rgba_regex.match(text)
        if rgb_match:
            text = [float(x) for x in rgb_match.groups() if x]
        self.colorSwatch.setColor(text)

    @Slot(np.ndarray)
    def _on_swatch_changed(self, color: np.ndarray):
        """Receive QColorSwatch change event, update the lineEdit, re-emit."""
        self.lineEdit.setText(color)
        self.color_changed.emit(color)


class QColorSwatch(QFrame):
    """A QFrame that displays a color.

    The internal color representation (self._color) is always a 1x4 np.ndarray
    """

    color_changed = Signal(np.ndarray)

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        tooltip: Optional[str] = None,
        initial_color: Optional[ColorType] = None,
    ):
        """Create a new ColorSwatch

        Parameters
        ----------
        parent : QWidget, optional
            parent widget, by default None
        tooltip : Optional[str], optional
            Tooltip when hovering on swatch,
            by default 'click to set color'
        initial_color : ColorType, optional
            initial color, by default will be transparent
        """
        super().__init__(parent)
        self.setObjectName('colorSwatch')
        self.setToolTip(tooltip or 'click to set color')

        self.color_changed.connect(self._update_swatch_style)
        self._color: np.ndarray = np.array([0, 0, 0, 0], np.float32)
        if initial_color is not None:
            self.setColor(initial_color)

    def color(self):
        """Return the current color"""
        return self._color

    def _update_swatch_style(self) -> None:
        """Convert the current color to rgba() string and update appearance."""
        col = f'rgba({",".join(map(lambda x: str(int(x*255)), self._color))})'
        self.setStyleSheet('#colorSwatch {background-color: ' + col + ';}')

    def mouseReleaseEvent(self, event):
        """Show QColorPopup picker when the user clicks on the swatch."""
        if event.button() == Qt.LeftButton:
            initial = QColor(*(255 * self._color).astype('int'))
            popup = QColorPopup(self, initial)
            popup.colorSelected.connect(self.setColor)
            popup.show_right_of_mouse()

    def setColor(self, color: ColorType) -> None:
        """Set the color of the swatch.

        Parameters
        ----------
        color : ColorType
            Can be any ColorType recognized by our
            utils.colormaps.standardize_color.transform_color function.
        """
        _color = transform_color(color)[0]
        emit = np.any(self._color != _color)
        self._color = _color
        if emit:
            self.color_changed.emit(_color)


class QColorLineEdit(QLineEdit):
    """A LineEdit that takes hex, rgb, or autocompletes common color names."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._compl = QCompleter(list(get_color_dict()))
        self.setCompleter(self._compl)
        self.setTextMargins(2, 2, 2, 2)

    def setText(self, color: ColorType):
        """Set the text of the lineEdit using any ColorType.

        Colors will be converted to standard SVG spec names if possible,
        or shown as #RGBA hex if not.

        Parameters
        ----------
        color : ColorType
            Can be any ColorType recognized by our
            utils.colormaps.standardize_color.transform_color function.
        """
        _rgb = transform_color(color)[0]
        _hex = rgb_to_hex(_rgb)[0]
        super().setText(hex_to_name.get(_hex, _hex))

    def event(self, event):
        """Pick the first name auto-completion upon pressing tab."""
        if event.type() == QEvent.KeyPress:
            if event.key() == Qt.Key_Tab:
                self._compl.popup().setCurrentIndex(self._compl.currentIndex())
                self.clearFocus()
                return True
        return super().event(event)


class CustomColorDialog(QColorDialog):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setObjectName('CustomColorDialog')

    def keyPressEvent(self, event):
        event.ignore()


class QColorPopup(QtPopup):
    """A QColorDialog inside of our QtPopup.

    Allows all of the show methods of QtPopup (like show relative to mouse).
    Passes through signals from the ColorDialogm, and handles some keypress
    events.
    """

    currentColorChanged = Signal(QColor)
    colorSelected = Signal(QColor)

    def __init__(self, parent=None, initial=None):
        super().__init__(parent)
        self.color_dialog = CustomColorDialog(self)

        # TODO: ask everyone how they feel about native vs QtColorDialog look
        # native dialog doesn't get added to the layout
        # so more would need to be done to use it
        self.color_dialog.setOptions(QColorDialog.DontUseNativeDialog)
        layout = QVBoxLayout()
        self.frame.setLayout(layout)
        layout.addWidget(self.color_dialog)

        self.setObjectName('QtColorPopup')
        self.color_dialog.currentColorChanged.connect(
            self.currentColorChanged.emit
        )
        self.color_dialog.colorSelected.connect(self._on_color_selected)
        self.color_dialog.rejected.connect(self._on_rejected)
        self.color_dialog.setCurrentColor(QColor(initial))

    def _on_color_selected(self, color):
        self.colorSelected.emit(color)
        self.close()

    def _on_rejected(self):
        self.close()

    def keyPressEvent(self, event):
        """Accept current color on enter, cancel on escape."""
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            return self.color_dialog.accept()
        if event.key() == Qt.Key_Escape:
            return self.color_dialog.reject()
        self.color_dialog.keyPressEvent(event)
