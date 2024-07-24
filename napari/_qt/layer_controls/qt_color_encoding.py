from __future__ import annotations

from abc import abstractmethod
from collections.abc import Iterable

import numpy as np
from qtpy.QtWidgets import (
    QComboBox,
    QFormLayout,
    QWidget,
)

from napari._qt.layer_controls.qt_colormap_combobox import QtColormapComboBox
from napari._qt.widgets.qt_color_swatch import QColorSwatchEdit
from napari.layers.utils.color_encoding import (
    ColorEncoding,
    ConstantColorEncoding,
    ManualColorEncoding,
    QuantitativeColorEncoding,
)
from napari.utils.colormaps.colormap_utils import AVAILABLE_COLORMAPS
from napari.utils.events.event import Event


class ColorEncodingWidget(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._layout = QFormLayout()
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(4)
        self._layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.setLayout(self._layout)

    @property
    @abstractmethod
    def model(self) -> ColorEncoding: ...


class ConstantColorEncodingWidget(ColorEncodingWidget):
    constant: QColorSwatchEdit

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._model = ConstantColorEncoding(constant='red')
        self.constant = QColorSwatchEdit(initial_color=self._model.constant)
        self.constant.color_changed.connect(self._onWidgetConstantChanged)
        self._layout.addRow(None, self.constant)

    @property
    def model(self) -> ColorEncoding:
        return self._model

    def setModel(self, model: ConstantColorEncoding) -> None:
        # TODO: disconnect old model?
        self._model = model
        self._model.events.constant.connect(self._onModelConstantChanged)
        self._setConstant(self._model.constant)

    def _onModelConstantChanged(self, event: Event) -> None:
        self._setConstant(event.value)

    def _setConstant(self, constant: np.ndarray) -> None:
        self.constant.setColor(constant)

    def _onWidgetConstantChanged(self, constant: np.ndarray) -> None:
        # TODO: need a blocker? or no change in value is enough?
        self._model.constant = constant


class ManualColorEncodingWidget(ColorEncodingWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._model = ManualColorEncoding(array=[])
        self.default = QColorSwatchEdit(initial_color=self._model.default)
        self.default.color_changed.connect(self._onWidgetDefaultChanged)
        self._layout.addRow('default', self.default)

    @property
    def model(self) -> ManualColorEncoding:
        return self._model

    def setModel(self, model: ManualColorEncoding) -> None:
        # TODO: disconnect old model?
        self._model = model
        self._model.events.default.connect(self._onModelDefaultChanged)
        self._setModelDefault(self._model.default)

    def _onModelDefaultChanged(self, event: Event) -> None:
        self._setModelDefault(event.value)

    def _setModelDefault(self, default: np.ndarray) -> None:
        self.default.setColor(default)

    def _onWidgetDefaultChanged(self, default: np.ndarray) -> None:
        self._model.default = default


class QuantitativeColorEncodingWidget(ColorEncodingWidget):
    feature: QComboBox
    colormap: QtColormapComboBox

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._model = QuantitativeColorEncoding(
            feature='',
            colormap='viridis',
        )
        self.feature = QComboBox()
        self.colormap = QtColormapComboBox(self)
        for name, cm in AVAILABLE_COLORMAPS.items():
            self.colormap.addItem(cm._display_name, name)
        self.colormap.currentTextChanged.connect(self._onWidgetColormapChanged)
        self.feature.currentTextChanged.connect(self._onWidgetFeatureChanged)
        self._layout.addRow('feature', self.feature)
        self._layout.addRow('colormap', self.colormap)

    @property
    def model(self) -> QuantitativeColorEncoding:
        return self._model

    def setModel(self, model: QuantitativeColorEncoding) -> None:
        # TODO: disconnect old model?
        self._model = model
        self._model.events.feature.connect(self._onModelFeatureChanged)
        self._model.events.colormap.connect(self._onModelColormapChanged)
        self.feature.setCurrentText(self._model.feature)
        self.colormap.setCurrentText(self._model.colormap.name)

    def setFeatures(self, features: Iterable[str]):
        # TODO: may need to block event.
        self.feature.clear()
        self.feature.addItems(features)

    def _onModelFeatureChanged(self, event: Event) -> None:
        feature = event.value
        self.feature.setCurrentText(feature)

    def _onModelColormapChanged(self, event: Event) -> None:
        # TODO: check what happens when specifying a custom colormap.
        colormap = event.value
        self.colormap.setCurrentText(colormap.name)

    def _onWidgetFeatureChanged(self, feature: str) -> None:
        self._model.feature = feature

    def _onWidgetColormapChanged(self, name: str):
        self._model.colormap = name
