from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QComboBox,
    QWidget,
)

from napari._qt.layer_controls.widgets.qt_widget_controls_base import (
    QtWidgetControlsBase,
    QtWrappedLabel,
)
from napari._qt.utils import qt_signals_blocked
from napari.layers import Vectors
from napari.layers.vectors._vectors_constants import VECTORSTYLE_TRANSLATIONS
from napari.utils.events.event_utils import connect_setattr
from napari.utils.translations import trans


class QtVectorStyleComboBoxControl(QtWidgetControlsBase):
    """
    Class that wraps the connection of events/signals between the layer vector style
    value attribute and Qt widgets.

    Parameters
    ----------
    parent: qtpy.QtWidgets.QWidget
        An instance of QWidget that will be used as widgets parent
    layer : napari.layers.Vectors
        An instance of a napari Vectors layer.

    Attributes
    ----------
    vector_style_combobox : qtpy.QtWidgets.QComboBox
        Dropdown widget to select vector_style for the vectors.
    vector_style_combobox_label : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for vector_style value chooser widget.
    """

    def __init__(self, parent: QWidget, layer: Vectors) -> None:
        super().__init__(parent, layer)
        # Setup layer
        self._layer.events.vector_style.connect(self._on_vector_style_change)

        # Setup widgets
        # dropdown to select the edge display vector_style
        vector_style_combobox = QComboBox(parent)
        for index, (data, text) in enumerate(VECTORSTYLE_TRANSLATIONS.items()):
            data = data.value
            vector_style_combobox.addItem(text, data)
            if data == self._layer.vector_style:
                vector_style_combobox.setCurrentIndex(index)

        self.vector_style_combobox = vector_style_combobox
        connect_setattr(
            self.vector_style_combobox.currentTextChanged,
            self._layer,
            'vector_style',
        )

        self.vector_style_combobox_label = QtWrappedLabel(
            trans._('vector style:')
        )

    def _on_vector_style_change(self) -> None:
        """Receive layer model vector style change event & update dropdown."""
        with qt_signals_blocked(self.vector_style_combobox):
            vector_style = self._layer.vector_style
            index = self.vector_style_combobox.findText(
                vector_style, Qt.MatchFlag.MatchFixedString
            )
            self.vector_style_combobox.setCurrentIndex(index)

    def get_widget_controls(self) -> list[tuple[QtWrappedLabel, QWidget]]:
        return [(self.vector_style_combobox_label, self.vector_style_combobox)]
