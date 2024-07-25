from __future__ import annotations

import logging
from typing import Any, Optional, Protocol, runtime_checkable

from qtpy.QtWidgets import (
    QComboBox,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from napari._qt.layer_controls.qt_color_encoding import (
    ColorEncodingWidget,
    ConstantColorEncodingWidget,
    DirectColorEncodingWidget,
    ManualColorEncodingWidget,
    QuantitativeColorEncodingWidget,
)
from napari.layers.utils.color_encoding import (
    ColorEncoding,
    ConstantColorEncoding,
    DirectColorEncoding,
    ManualColorEncoding,
    QuantitativeColorEncoding,
)
from napari.layers.utils.style_encoding import StyleCollection
from napari.utils.compat import StrEnum
from napari.utils.events.event import Event, EventEmitter


class ColorMode(StrEnum):
    CONSTANT = 'constant'
    DIRECT = 'direct'
    MANUAL = 'manual'
    QUANTITATIVE = 'quantitative'


@runtime_checkable
class StyledLayerEvents(Protocol):
    features: EventEmitter
    style: EventEmitter


@runtime_checkable
class StyledLayer(Protocol):
    features: Any
    style: StyleCollection
    events: StyledLayerEvents


class ColorControlsWidget(QWidget):
    """Controls color encoding associated with a layer style property.

    This provides an abstraction on top of the color encoding widgets
    to allow switching between different encoding types in the napari
    GUI.

    If an encoding is not recognized (e.g. it's a custom implementation
    of the encoding protocol), then this widget will inform the user,
    but will offer no controls.
    """

    layer: StyledLayer
    attr: str
    customLabel: QLabel
    mode: QComboBox
    encodings: dict[ColorMode, ColorEncodingWidget]

    def __init__(
        self,
        layer: StyledLayer,
        attr: str,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent=parent)

        self.layer = layer
        self.attr = attr

        # TODO: disconnect?
        # Or will be there be one widget instance per layer instance?
        self.layer.events.features.connect(self._onLayerFeaturesChanged)
        getattr(self.layer.style.events, attr).connect(
            self._onLayerEncodingChanged
        )

        # Shown when encoding type is not supported.
        self.customLabel = QLabel('Custom')

        self.mode = QComboBox(self)
        self.mode.addItems(tuple(ColorMode))
        self.mode.setCurrentText('')
        self.mode.currentTextChanged.connect(self._onModeChanged)

        # Always create every type of widget so that we have strong typing.
        # TODO: or create each encoding on demand.
        self._constant = ConstantColorEncodingWidget()
        self._direct = DirectColorEncodingWidget()
        self._manual = ManualColorEncodingWidget()
        self._quantitative = QuantitativeColorEncodingWidget()

        self.encodings: dict[ColorMode, ColorEncodingWidget] = {
            ColorMode.CONSTANT: self._constant,
            ColorMode.DIRECT: self._direct,
            ColorMode.MANUAL: self._manual,
            ColorMode.QUANTITATIVE: self._quantitative,
        }

        # TODO: need to ensure that the layer's current model instance is used.
        # Probably better to create, connect/disconnect specific encoding
        # widgets as the layer encoding and mode changes.
        # That way handling the instances and fields of specific encoding types
        # can be done much more directly.
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self.mode)
        for encoding in self.encodings.values():
            layout.addWidget(encoding)
        self.setLayout(layout)

        self._setFeatures(layer.features)
        self._setCurrentEncoding(getattr(layer.style, attr))

    def _onLayerFeaturesChanged(self, event: Event) -> None:
        self._setFeatures(event.value)

    def _setFeatures(self, features: Any) -> None:
        self._direct.setFeatures(features.columns)
        self._quantitative.setFeatures(features.columns)

    def _onLayerEncodingChanged(self, event: Event) -> None:
        self._setCurrentEncoding(event.value)

    def _setCurrentEncoding(self, currentEncoding: ColorEncoding) -> None:
        logging.warning('_onLayerEncodingChanged: %s', currentEncoding)
        if isinstance(currentEncoding, ConstantColorEncoding):
            self._constant.setModel(currentEncoding)
            self.mode.setCurrentText('constant')
        elif isinstance(currentEncoding, DirectColorEncoding):
            self._direct.setModel(currentEncoding)
            self.mode.setCurrentText('direct')
        elif isinstance(currentEncoding, ManualColorEncoding):
            self._manual.setModel(currentEncoding)
            self.mode.setCurrentText('manual')
        elif isinstance(currentEncoding, QuantitativeColorEncoding):
            self._quantitative.setModel(currentEncoding)
            self.mode.setCurrentText('quantitative')
        else:
            logging.warning('Unsupported color encoding: %s', currentEncoding)
            self._onModeChanged('')
            self.customLabel.show()

    def _onModeChanged(self, mode: str) -> None:
        logging.warning('_onModeChanged: %s', mode)
        selected = (
            self.encodings[ColorMode(mode)]
            if mode in tuple(ColorMode)
            else None
        )
        for encoding in self.encodings.values():
            encoding.setVisible(encoding is selected)
        if selected is not None:
            self._updateLayerModel(selected.model)

    def _updateLayerModel(self, model: ColorEncoding) -> None:
        setattr(self.layer.style, self.attr, model)
