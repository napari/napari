from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from qtpy.QtWidgets import (
    QDoubleSpinBox,
    QWidget,
)

from napari._qt.layer_controls.dynamic.widgets.qt_widget_controls_base import (
    QtWidgetControlsBase,
    QtWrappedLabel,
)
from napari._qt.utils import attr_to_settr

if TYPE_CHECKING:
    from napari.layers import Vectors


class QtWidthSpinBoxControl(QtWidgetControlsBase):
    """
    Class that wraps the connection of events/signals between the layer edge width
    attribute and Qt widgets.

    Parameters
    ----------
    parent: qtpy.QtWidgets.QWidget
        An instance of QWidget that will be used as widgets parent
    layers : list[napari.layers.Vectors]
        A list of napari Vectors layers.

    Attributes
    ----------
    width_spinbox : qtpy.QtWidgets.QDoubleSpinBox
        Spin box widget controlling edge width of vectors.
    width_spinbox_label : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for the edge width of vectors chooser widget.
    """

    _layers: list[Vectors]

    def __init__(
        self, layers: list[Vectors], parent: QWidget | None = None
    ) -> None:
        super().__init__(layers, parent)
        # Setup widgets
        # line width in pixels
        self.width_spinbox = QDoubleSpinBox()
        self.width_spinbox.setKeyboardTracking(False)
        self.width_spinbox.setSingleStep(0.1)
        self.width_spinbox.setMinimum(0.01)
        self.width_spinbox.setMaximum(np.inf)
        self.width_spinbox.setValue(self._layers[0].edge_width)
        self.width_spinbox.valueChanged.connect(self.change_width)
        self._callbacks.append(
            attr_to_settr(
                self._layers[0],
                'edge_width',
                self.width_spinbox,
                'setValue',
            )
        )
        self.width_spinbox_label = QtWrappedLabel('width:')

    def change_width(self, value) -> None:
        """Change edge line width of vectors on the layer model.
        Parameters
        ----------
        value : float
            Line width of vectors.
        """
        for layer in self._layers:
            layer.edge_width = value
        self.width_spinbox.clearFocus()
        # TODO: Check other way to give focus without calling parent
        if self.parent() is not None:
            self.parent().setFocus()

    def get_widget_controls(
        self,
    ) -> list[tuple[QtWrappedLabel, QWidget] | tuple[QWidget]]:
        return [(self.width_spinbox_label, self.width_spinbox)]


class QtLengthSpinBoxControl(QtWidgetControlsBase):
    """
    Class that wraps the connection of events/signals between the layer length of vectors
    attribute and Qt widgets.

    Parameters
    ----------
    parent: qtpy.QtWidgets.QWidget
        An instance of QWidget that will be used as widgets parent
    layer : napari.layers.Vectors
        An instance of a napari Vectors layer.

    Attributes
    ----------
    length_spinbox : qtpy.QtWidgets.QDoubleSpinBox
        Spinbox widget controlling line length of vectors.
        Multiplicative factor on projections for length of all vectors.
    length_spinbox_label : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for line length of vectors chooser widget.
    """

    _layers: list[Vectors]

    def __init__(
        self, layers: list[Vectors], parent: QWidget | None = None
    ) -> None:
        super().__init__(layers, parent)

        # Setup widgets
        # line length
        self.length_spinbox = QDoubleSpinBox()
        self.length_spinbox.setKeyboardTracking(False)
        self.length_spinbox.setSingleStep(0.1)
        self.length_spinbox.setValue(self._layers[0].length)
        self.length_spinbox.setMinimum(0.1)
        self.length_spinbox.setMaximum(np.inf)
        self.length_spinbox.valueChanged.connect(self.change_length)
        self._callbacks.append(
            attr_to_settr(
                self._layers[0],
                'length',
                self.length_spinbox,
                'setValue',
            )
        )
        self.length_spinbox_label = QtWrappedLabel('length:')

    def change_length(self, value: float) -> None:
        """Change length of vectors on the layer model.
        Multiplicative factor on projections for length of all vectors.
        Parameters
        ----------
        value : float
            Length of vectors.
        """
        for layer in self._layers:
            layer.length = value
        self.length_spinbox.clearFocus()
        # TODO: Check other way to give focus without calling parent
        if self.parent() is not None:
            self.parent().setFocus()

    def get_widget_controls(
        self,
    ) -> list[tuple[QtWrappedLabel, QWidget] | tuple[QWidget]]:
        return [(self.length_spinbox_label, self.length_spinbox)]
