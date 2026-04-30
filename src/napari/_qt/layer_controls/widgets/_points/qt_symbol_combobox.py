from qtpy.QtWidgets import QWidget
from superqt import QEnumComboBox

from napari._qt.layer_controls.widgets.qt_widget_controls_base import (
    QtWidgetControlsBase,
    QtWrappedLabel,
)
from napari._qt.utils import qt_signals_blocked
from napari.layers import Points
from napari.layers.points._points_constants import Symbol


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
        sym_cb = QEnumComboBox(enum_class=Symbol)
        sym_cb.setToolTip(
            'Change the symbol of currently selected points and any added afterwards.'
        )
        sym_cb.setCurrentEnum(self._layer.current_symbol)
        sym_cb.currentEnumChanged.connect(self.change_current_symbol)
        self.symbol_combobox = sym_cb

        self.symbol_combobox_label = QtWrappedLabel('symbol:')

    def change_current_symbol(self, symbol: Symbol) -> None:
        """Change marker symbol of the points on the layer model.

        Parameters
        ----------
        text : int
            Index of current marker symbol of points, eg: '+', '.', etc.
        """
        with self._layer.events.symbol.blocker(self._on_current_symbol_change):
            self._layer.current_symbol = symbol

    def _on_current_symbol_change(self) -> None:
        """Receive marker symbol change event and update the dropdown menu."""
        with qt_signals_blocked(self.symbol_combobox):
            self.symbol_combobox.setCurrentEnum(self._layer.current_symbol)

    def get_widget_controls(self) -> list[tuple[QtWrappedLabel, QWidget]]:
        return [(self.symbol_combobox_label, self.symbol_combobox)]
