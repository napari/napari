import fastplotlib as fpl
from napari.components import ViewerModel
from PyQt5 import QtWidgets
from wgpu.gui.qt import WgpuWidget, WgpuCanvas
import wgpu.backends.rs

import pygfx as gfx
import numpy as np

# TODO Figure out how to get fastplotlib canvas into Qt application and not just as a docked widget


class FastplotlibCanvas(QtWidgets.QWidget):
    """
    Fastplotlib canvas class.
    """

    def __init__(
            self,
    ) -> None:
        super().__init__(None)
        self._canvas = WgpuCanvas(parent=self)
        self._renderer = gfx.WgpuRenderer(self._canvas)
        self._plot = fpl.Plot(
            canvas=self._canvas,
            renderer=self._renderer,
        )

        # create a layout
        layout = QtWidgets.QHBoxLayout()
        self.setLayout(layout)
        # add canvas to Widget
        layout.addWidget(self._canvas)
        self._plot.canvas.setFixedSize(400, 400)

        # start rendering process
        self._plot.canvas.request_draw(self._plot.render)

    def add_layer_visual_mapping(self, napari_layer, fpl_layer):
        """Maps a napari layer to its corresponding fastplotlib layer.
        Paremeters
        ----------
        napari_layer : napari.layers
           Any napari layer, the layer type is the same as the vispy layer.
        fpl_layer : napari._fastplotlib.layers
           Any fastplotlib layer, the layer type is the same as the napari layer.

        Returns
        -------
        None
        """
        self.layer_to_visual[napari_layer] = fpl_layer

        self._plot.add_graphic(fpl_layer.image_graphic)

      #  napari_layer.events.visible.connect(self._reorder_layers)
     #   self.viewer.camera.events.angles.connect(fpl_layer._on_camera_move)

    #    self._reorder_layers()



