from napari._qt.layers.qt_image_base_layer import QtBaseImageControls
from napari._vispy.vispy_base_layer import VispyBaseLayer
from napari.layers import Layer


class ControllerBase:
    """
    Base layer controller class responsible for the interactions between vispy, qt and the data
    """

    def __init__(
        self,
        layer: Layer,
        qt_controls: QtBaseImageControls,
        vispy_layer: VispyBaseLayer,
    ):
        """
        Parameters
        ----------
        layer: Layer
            The data layer
        qt_controls: QtBaseImageControls
            The qt_controls attached to the layer
        vispy_layer: VispyBaseLayer
            The vispy rendering of the layer
        """

        self.layer = layer
        self.qt_controls = qt_controls
        self.vispy_layer = vispy_layer
