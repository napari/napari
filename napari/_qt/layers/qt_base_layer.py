from qtpy.QtCore import Qt
from qtpy.QtWidgets import QSlider, QGridLayout, QFrame, QComboBox

from ...layers.base._base_layer_interface import BaseLayerInterface
from ...layers.base._base_constants import Blending
from ...utils.event import Event, EmitterGroup


class QtLayerControls(QFrame, BaseLayerInterface):
    """Superclass for all the other LayerControl classes.

    This class is never directly instantiated anywhere.

    Parameters
    ----------
    layer : napari.layers.Layer
        An instance of a napari layer.

    Attributes
    ----------
    blendComboBox : qtpy.QtWidgets.QComboBox
        Drowpdown widget to select blending mode of layer.
    grid_layout : qtpy.QtWidgets.QGridLayout
        Layout of Qt widget controls for the layer.
    layer : napari.layers.Layer
        An instance of a napari layer.
    opacitySlider : qtpy.QtWidgets.QSlider
        Slider controlling opacity of the layer.
    """

    def __init__(self, layer):
        super().__init__()

        self.layer = layer

        self.layer.event_handler.register_component_to_update(self)
        self.events = EmitterGroup(
            source=self,
            blending=Event,
            opacity=Event,
            event_handler_callback=self.layer.event_handler.on_change,
        )
        self.setObjectName('layer')
        self.setMouseTracking(True)

        self.grid_layout = QGridLayout(self)
        self.grid_layout.setContentsMargins(0, 0, 0, 0)
        self.grid_layout.setSpacing(2)
        self.grid_layout.setColumnMinimumWidth(0, 86)
        self.grid_layout.setColumnStretch(1, 1)
        self.setLayout(self.grid_layout)

        sld = QSlider(Qt.Horizontal, parent=self)
        sld.setFocusPolicy(Qt.NoFocus)
        sld.setMinimum(0)
        sld.setMaximum(100)
        sld.setSingleStep(1)

        self.emit_opacity_event = lambda value: self.events.opacity(
            value=value / 100
        )
        sld.valueChanged.connect(self.emit_opacity_event)
        self.opacitySlider = sld
        self._on_opacity_change(self.layer.opacity)

        blend_comboBox = QComboBox(self)
        blend_comboBox.addItems(Blending.keys())
        index = blend_comboBox.findText(
            self.layer.blending, Qt.MatchFixedString
        )
        blend_comboBox.setCurrentIndex(index)
        blend_comboBox.activated[str].connect(self.events.blending)
        self.blendComboBox = blend_comboBox

    def _on_opacity_change(self, value):
        """Receive layer model opacity change event and update opacity slider.

        Parameters
        ----------
        value : int
            opacity value to change set the slider to.
        """
        self.opacitySlider.setValue(value * 100)

    def _on_blending_change(self, value):
        """Receive layer model blending mode change event and update slider.

        Parameters
        ----------
        value : str
           The value to set the blending element to
        """
        index = self.blendComboBox.findText(value, Qt.MatchFixedString)
        self.blendComboBox.setCurrentIndex(index)
