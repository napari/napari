from __future__ import annotations

import logging
from abc import abstractmethod
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
from psygnal import Signal
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QVBoxLayout,
    QWidget,
)

from napari._qt.layer_controls.qt_colormap_combobox import QtColormapComboBox
from napari._qt.layer_controls.qt_layer_controls_base import QtLayerControls
from napari._qt.widgets.qt_color_swatch import QColorSwatchEdit
from napari.layers.base._base_constants import Mode
from napari.layers.utils.color_encoding import (
    ColorEncoding,
    ConstantColorEncoding,
    ManualColorEncoding,
    QuantitativeColorEncoding,
)
from napari.layers.vectors._vectors_constants import VECTORSTYLE_TRANSLATIONS
from napari.utils.colormaps.colormap_utils import AVAILABLE_COLORMAPS
from napari.utils.compat import StrEnum
from napari.utils.translations import trans

if TYPE_CHECKING:
    import napari.layers


class ColorMode(StrEnum):
    CONSTANT = 'constant'
    MANUAL = 'manual'
    QUANTITATIVE = 'quantitative'


class ColorModeWidget(QWidget):
    def __init__(
        self,
        layer: napari.layers.Vectors,
        attr: str,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent=parent)

        self.layer = layer
        self.attr = attr

        # TODO: disconnect this?
        self.layer.events.features.connect(self._onLayerFeaturesChanged)

        getattr(self.layer.style.events, attr).connect(
            self._onLayerEncodingChanged
        )

        self.mode = QComboBox(self)
        self.mode.addItems(tuple(ColorMode))
        self.mode.setCurrentText('')
        self.mode.currentTextChanged.connect(self._onModeChanged)

        self._constant = ConstantColorEncodingWidget()
        self._manual = ManualColorEncodingWidget()
        self._quantitative = QuantitativeColorEncodingWidget()

        self.encodings: dict[ColorMode, ColorEncodingWidget] = {
            ColorMode.CONSTANT: self._constant,
            ColorMode.MANUAL: self._manual,
            ColorMode.QUANTITATIVE: self._quantitative,
        }

        for encoding in self.encodings.values():
            encoding.modelChanged.connect(self._updateLayerModel)

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

        self._onLayerFeaturesChanged(layer.features)
        self._onLayerEncodingChanged(getattr(layer.style, attr))
        self._onModeChanged(self.mode.currentText())

    def _onLayerEncodingChanged(self, currentEncoding: ColorEncoding) -> None:
        logging.warning('_onLayerEncodingChanged: %s', currentEncoding)
        for mode, encoding in self.encodings.items():
            if isinstance(currentEncoding, type(encoding.model)):
                self.encodings[mode]._model = currentEncoding
                self.mode.setCurrentText(str(mode))

    def _onLayerFeaturesChanged(self, features: Any) -> None:
        self._quantitative.setFeatures(features.columns)

    def _onModeChanged(self, mode: str) -> None:
        logging.warning('_onModeChanged: %s', mode)
        selected = self.encodings[ColorMode(mode)]
        for encoding in self.encodings.values():
            encoding.setVisible(encoding is selected)
            if encoding is selected:
                self._updateLayerModel(encoding.model)

    def _updateLayerModel(self, model: ColorEncoding) -> None:
        setattr(self.layer.style, self.attr, model)


class ColorEncodingWidget(QWidget):
    modelChanged = Signal(ColorEncoding)

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

    def setModel(self, model: ColorEncoding) -> None:
        self._model = model
        self.modelChanged(self._model)


class ConstantColorEncodingWidget(ColorEncodingWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._model = ConstantColorEncoding(constant='red')
        self.constant = QColorSwatchEdit(initial_color=self._model.constant)
        self.constant.color_changed.connect(self._onConstantChanged)
        self._layout.addRow(None, self.constant)

    @property
    def model(self) -> ColorEncoding:
        return self._model

    def _onConstantChanged(self, constant: np.ndarray) -> None:
        self.setModel(ConstantColorEncoding(constant=constant))


class ManualColorEncodingWidget(ColorEncodingWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._model = ManualColorEncoding(array=[])
        self.default = QColorSwatchEdit(initial_color=self._model.default)
        self.default.color_changed.connect(self._onDefaultChanged)
        self._layout.addRow('default', self.default)

    @property
    def model(self) -> ManualColorEncoding:
        return self._model

    def _onDefaultChanged(self, default: np.ndarray) -> None:
        self.setModel(
            ManualColorEncoding(
                array=self._model.array,
                default=default,
            )
        )


class QuantitativeColorEncodingWidget(ColorEncodingWidget):
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
        self.colormap.currentTextChanged.connect(self._onColormapChanged)
        self.feature.currentTextChanged.connect(self._onFeatureChanged)

        self._layout.addRow('feature', self.feature)
        self._layout.addRow('colormap', self.colormap)

    @property
    def model(self) -> QuantitativeColorEncoding:
        return self._model

    def setFeatures(self, features: Iterable[str]):
        # TODO: may need to block event.
        self.feature.clear()
        self.feature.addItems(features)

    def _onFeatureChanged(self, feature: str) -> None:
        self.setModel(
            QuantitativeColorEncoding(
                feature=feature,
                colormap=self._model.colormap,
            )
        )

    def _onColormapChanged(self, name: str):
        self.setModel(
            QuantitativeColorEncoding(
                feature=self._model.feature,
                colormap=name,
            )
        )


class QtVectorsControls(QtLayerControls):
    """Qt view and controls for the napari Vectors layer.

    Parameters
    ----------
    layer : napari.layers.Vectors
        An instance of a napari Vectors layer.

    Attributes
    ----------
    layer : napari.layers.Vectors
        An instance of a napari Vectors layer.
    MODE : Enum
        Available modes in the associated layer.
    PAN_ZOOM_ACTION_NAME : str
        String id for the pan-zoom action to bind to the pan_zoom button.
    TRANSFORM_ACTION_NAME : str
        String id for the transform action to bind to the transform button.
    button_group : qtpy.QtWidgets.QButtonGroup
        Button group of points layer modes (ADD, PAN_ZOOM, SELECT).
    panzoom_button : napari._qt.widgets.qt_mode_button.QtModeRadioButton
        Button for pan/zoom mode.
    transform_button : napari._qt.widgets.qt_mode_button.QtModeRadioButton
        Button to select transform mode.
    edgeColor : ColorModeWidget
        Controls layer.style.edge_color property.
    vector_style_comboBox : qtpy.QtWidgets.QComboBox
        Dropdown widget to select vector_style for the vectors.
    layer : napari.layers.Vectors
        An instance of a napari Vectors layer.
    outOfSliceCheckBox : qtpy.QtWidgets.QCheckBox
        Checkbox to indicate whether to render out of slice.
    lengthSpinBox : qtpy.QtWidgets.QDoubleSpinBox
        Spin box widget controlling line length of vectors.
        Multiplicative factor on projections for length of all vectors.
    widthSpinBox : qtpy.QtWidgets.QDoubleSpinBox
        Spin box widget controlling edge line width of vectors.
    vector_style_comboBox : qtpy.QtWidgets.QComboBox
        Dropdown widget to select vector_style for the vectors.
    """

    layer: napari.layers.Vectors
    MODE = Mode
    PAN_ZOOM_ACTION_NAME = 'activate_tracks_pan_zoom_mode'
    TRANSFORM_ACTION_NAME = 'activate_tracks_transform_mode'

    def __init__(self, layer) -> None:
        super().__init__(layer)

        self.edgeColor = ColorModeWidget(layer, attr='edge_color')

        # dropdown to select the edge display vector_style
        vector_style_comboBox = QComboBox(self)
        for index, (data, text) in enumerate(VECTORSTYLE_TRANSLATIONS.items()):
            data = data.value
            vector_style_comboBox.addItem(text, data)
            if data == self.layer.vector_style:
                vector_style_comboBox.setCurrentIndex(index)

        self.vector_style_comboBox = vector_style_comboBox
        self.vector_style_comboBox.currentTextChanged.connect(
            self.change_vector_style
        )

        # line width in pixels
        self.widthSpinBox = QDoubleSpinBox()
        self.widthSpinBox.setKeyboardTracking(False)
        self.widthSpinBox.setSingleStep(0.1)
        self.widthSpinBox.setMinimum(0.1)
        self.widthSpinBox.setMaximum(np.inf)
        self.widthSpinBox.setValue(self.layer.edge_width)
        self.widthSpinBox.valueChanged.connect(self.change_width)

        # line length
        self.lengthSpinBox = QDoubleSpinBox()
        self.lengthSpinBox.setKeyboardTracking(False)
        self.lengthSpinBox.setSingleStep(0.1)
        self.lengthSpinBox.setValue(self.layer.length)
        self.lengthSpinBox.setMinimum(0.1)
        self.lengthSpinBox.setMaximum(np.inf)
        self.lengthSpinBox.valueChanged.connect(self.change_length)

        out_of_slice_cb = QCheckBox()
        out_of_slice_cb.setToolTip(trans._('Out of slice display'))
        out_of_slice_cb.setChecked(self.layer.out_of_slice_display)
        out_of_slice_cb.stateChanged.connect(self.change_out_of_slice)
        self.outOfSliceCheckBox = out_of_slice_cb

        self.layout().addRow(self.button_grid)
        self.layout().addRow(self.opacityLabel, self.opacitySlider)
        self.layout().addRow(trans._('width:'), self.widthSpinBox)
        self.layout().addRow(trans._('length:'), self.lengthSpinBox)
        self.layout().addRow(trans._('blending:'), self.blendComboBox)
        self.layout().addRow(
            trans._('vector style:'), self.vector_style_comboBox
        )
        self.layout().addRow(trans._('edge color:'), self.edgeColor)
        self.layout().addRow(trans._('out of slice:'), self.outOfSliceCheckBox)

        self.layer.events.edge_width.connect(self._on_edge_width_change)
        self.layer.events.length.connect(self._on_length_change)
        self.layer.events.out_of_slice_display.connect(
            self._on_out_of_slice_display_change
        )
        self.layer.events.vector_style.connect(self._on_vector_style_change)
        # self.layer.events.edge_color.connect(self._on_edge_color_change)

    def change_vector_style(self, vector_style: str):
        """Change vector style of vectors on the layer model.

        Parameters
        ----------
        vector_style : str
            Name of vectors style, eg: 'line', 'triangle' or 'arrow'.
        """
        with self.layer.events.vector_style.blocker():
            self.layer.vector_style = vector_style

    def change_width(self, value):
        """Change edge line width of vectors on the layer model.

        Parameters
        ----------
        value : float
            Line width of vectors.
        """
        self.layer.edge_width = value
        self.widthSpinBox.clearFocus()
        self.setFocus()

    def change_length(self, value):
        """Change length of vectors on the layer model.

        Multiplicative factor on projections for length of all vectors.

        Parameters
        ----------
        value : float
            Length of vectors.
        """
        self.layer.length = value
        self.lengthSpinBox.clearFocus()
        self.setFocus()

    def change_out_of_slice(self, state):
        """Toggle out of slice display of vectors layer.

        Parameters
        ----------
        state : int
            Integer value of Qt.CheckState that indicates the check state of outOfSliceCheckBox
        """
        self.layer.out_of_slice_display = (
            Qt.CheckState(state) == Qt.CheckState.Checked
        )

    def _on_length_change(self):
        """Change length of vectors."""
        with self.layer.events.length.blocker():
            self.lengthSpinBox.setValue(self.layer.length)

    def _on_out_of_slice_display_change(self, event):
        """Receive layer model out_of_slice_display change event and update checkbox."""
        with self.layer.events.out_of_slice_display.blocker():
            self.outOfSliceCheckBox.setChecked(self.layer.out_of_slice_display)

    def _on_edge_width_change(self):
        """Receive layer model width change event and update width spinbox."""
        with self.layer.events.edge_width.blocker():
            self.widthSpinBox.setValue(self.layer.edge_width)

    def _on_vector_style_change(self):
        """Receive layer model vector style change event & update dropdown."""
        with self.layer.events.vector_style.blocker():
            vector_style = self.layer.vector_style
            index = self.vector_style_comboBox.findText(
                vector_style, Qt.MatchFixedString
            )
            self.vector_style_comboBox.setCurrentIndex(index)
