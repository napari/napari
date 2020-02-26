from ._base import ControllerBase

from ..layers.image.image import Image
from .._qt.layers.qt_image_layer import QtImageControls
from .._vispy.vispy_image_layer import VispyImageLayer


class ImageController(ControllerBase):
    def __init__(
        self,
        layer: Image,
        qt_controls: QtImageControls,
        vispy_layer: VispyImageLayer,
    ):
        super().__init__(layer, qt_controls, vispy_layer)

        # connect to specific image layer events
        self.layer.events.interpolation.connect(self.on_interpolation_change)

        # connect to qt events
        self.qt_controls.events.interpolation.connect(
            self.on_interpolation_change
        )

    def on_interpolation_change(self, event=None):
        """
        Process changes when the interpolation attribute is changed from any interface
        """
        value = event.interpolation
        # update layer
        self.layer._set_interpolation(value)
        # update qt
        self.qt_controls.set_interpolation(value)
        # update vispy
        self.vispy_layer.set_interpolation(value)
