from qtpy.QtWidgets import QComboBox, QWidget

from napari._qt.layer_controls.widgets.qt_widget_controls_base import (
    QtWidgetControlsBase,
    QtWrappedLabel,
)
from napari._qt.utils import qt_signals_blocked
from napari.layers import Points
from napari.layers.points._points_constants import (
    SYMBOL_TRANSLATION,
    SYMBOL_TRANSLATION_INVERTED,
)
from napari.utils.translations import trans


class QtSymbolComboBoxControl(QtWidgetControlsBase):
    """
    Class that wraps the connection of events/signals between the current symbol
    layer attribute and Qt widgets.

    Parameters
    ----------
    parent: qtpy.QtWidgets.QWidget
        An instance of QWidget that will be used as widgets parent
    layer : napari.layers.Points
        An instance of a napari Points layer.

    Attributes
    ----------
    symbol_combobox : qtpy.QtWidgets.QComboBox
        Combobox controlling current symbol attribute of the layer.
    symbol_combobox_label : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for the current symbol chooser widget.
    """

    def __init__(self, parent: QWidget, layer: Points) -> None:
        super().__init__(parent, layer)
        # Setup layer
        self._layer.events.symbol.connect(self._on_current_symbol_change)
        self._layer.events.current_symbol.connect(
            self._on_current_symbol_change
        )

        # Setup widgets
        sym_cb = QComboBox()
        sym_cb.setToolTip(
            trans._(
                'Change the symbol of currently selected points and any added afterwards.'
            )
        )
        current_index = 0
        for index, (symbol_string, text) in enumerate(
            SYMBOL_TRANSLATION.items()
        ):
            symbol_string = symbol_string.value
            sym_cb.addItem(text, symbol_string)

            if symbol_string == self._layer.current_symbol:
                current_index = index

        sym_cb.setCurrentIndex(current_index)
        sym_cb.currentTextChanged.connect(self.change_current_symbol)
        self.symbol_combobox = sym_cb

        self.symbol_combobox_label = QtWrappedLabel(trans._('symbol:'))

    def change_current_symbol(self, text: str) -> None:
        """Change marker symbol of the points on the layer model.

        Parameters
        ----------
        text : int
            Index of current marker symbol of points, eg: '+', '.', etc.
        """
        with self._layer.events.symbol.blocker(self._on_current_symbol_change):
            self._layer.current_symbol = SYMBOL_TRANSLATION_INVERTED[text]

    def _on_current_symbol_change(self) -> None:
        """Receive marker symbol change event and update the dropdown menu."""
        with qt_signals_blocked(self.symbol_combobox):
            self.symbol_combobox.setCurrentIndex(
                self.symbol_combobox.findData(self._layer.current_symbol.value)
            )

    def get_widget_controls(self) -> list[tuple[QtWrappedLabel, QWidget]]:
        return [(self.symbol_combobox_label, self.symbol_combobox)]
