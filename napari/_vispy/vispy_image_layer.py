#from qtpy.QtWidgets import QHBoxLayout
#from . import QVRangeSlider
#from .qt_base_layer import QtLayerControls, QtLayerProperties
#from qtpy.QtCore import Qt
#from qtpy.QtWidgets import QLabel, QComboBox
#from ..layers.image._constants import Interpolation
from vispy.scene.visuals import Image as ImageNode
from .vispy_base_layer import VispyBaseLayer


class VispyImageLayer(VispyBaseLayer):
    def __init__(self, layer):
        node = ImageNode(None, method='auto')
        super().__init__(layer, node)

        self.layer.events.interpolation.connect(lambda e: self._on_interpolation_change())

    def _on_opacity_change(self):
        self.node.interpolation = self.layer.interpolation

    def _on_data_change(self):
        self.node.set_data(self.layer._data_view)
        self.node.update()

    def on_mouse_move(self, event):
        """Called whenever mouse moves over canvas."""
        if event.pos is None:
            return
        #self.position = tuple(event.pos)
        #coord, value = self.get_value()
        #self.status = self.get_message(coord, value)
