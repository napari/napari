from .qt import QtViewer
from copy import copy
from numpy import clip, integer, ndarray


class Viewer:
    """Viewer containing the rendered scene, layers, and controlling elements
    including dimension sliders.

    Parameters
    ----------
    parent : Window
        Parent window.

    Attributes
    ----------
    gui : Gui
        Parent gui window.
    layers : LayerList
        List of contained layers.
    dimensions : Dimensions
        Contains axes, indices, dimensions and sliders.
    controlBars : ControlBars
        Contains contorl bar sliders.
    """
    def __init__(self, gui):
        from ._layer_list import LayerList
        from ._control_bars import ControlBars
        from ._center import Center

        self.gui = gui
        self.dimensions = Center(self)
        self.layers = LayerList(self)
        self.controlBars = ControlBars()
        #self.canvas = None
        #self.view = None
        self._qt = QtViewer(self)
        self._view = self.dimensions._view
        self._canvas = self.dimensions._canvas
        self._update = self.dimensions._update
        self._statusBar = self.gui._qt_window.statusBar

    def add_layer(self, layer):
        """Adds a layer to the viewer.

        Parameters
        ----------
        layer : Layer
            Layer to add.
        """
        self.layers.append(layer)
        if len(self.layers) == 1:
            self.dimensions.reset_view()

    def imshow(self, image, meta=None, multichannel=None, **kwargs):
        """Shows an image in the viewer.

        Parameters
        ----------
        image : np.ndarray
            Image data.
        meta : dict, optional
            Image metadata.
        multichannel : bool, optional
            Whether the image is multichannel. Guesses if None.
        **kwargs : dict
            Parameters that will be translated to metadata.

        Returns
        -------
        layer : Image
            Layer for the image.
        """
        meta = guess_metadata(image, meta, multichannel, kwargs)

        return self.add_image(image, meta)

    def _update_layers(self):
        """Updates the contained layers.
        """
        for layer in self.layers:
            layer._set_view_slice(self.dimensions.indices)

        self.update_statusBar()

    def _on_layers_change(self, event):
        """Called whenever a layer is changed.
        """
        self.dimensions._child_layer_changed = True
        self.dimensions._update()

    def update_statusBar(self):
        from ..layers._image_layer import Image
        from ..layers._markers_layer import Markers

        msg = '(%d, %d' % (self.dimensions._pos[0], self.dimensions._pos[1])
        if self.dimensions.max_dims > 2:
            for i in range(2,self.dimensions.max_dims):
                msg = msg + ', %d' % self.dimensions.indices[i]
        msg = msg + ')'

        top_markers = []
        for i, layer in enumerate(self.layers[::-1]):
            if layer.visible and isinstance(layer, Image):
                top_image = len(self.layers) - 1 - i
                break
            elif layer.visible and isinstance(layer, Markers):
                top_markers.append(len(self.layers) - 1 - i)
        else:
            top_image = None

        index = None
        for i in top_markers:
            indices = copy(self.dimensions.indices)
            indices[0] = int(self.dimensions._pos[1])
            indices[1] = int(self.dimensions._pos[0])
            index = self.layers[i]._selected_markers(indices)
            if index is None:
                pass
            else:
                msg = msg + ', index %d, layer %d' % (index, i)
                break

        if top_image is None:
            pass
        elif index is None:
            indices = copy(self.dimensions.indices)
            indices[0] = int(self.dimensions._pos[0])
            indices[1] = int(self.dimensions._pos[1])
            value = self.layers[top_image]._slice_image(indices)
            msg = msg + ', value '
            if isinstance(value, ndarray):
                if isinstance(value[0], integer):
                    msg = msg + '(%d, %d, %d)' % (value[0], value[1], value[2])
                else:
                    msg = msg + '(%.3f, %.3f, %.3f)' % (value[0], value[1], value[2])
            else:
                if isinstance(value, integer):
                    msg = msg + '%d' % value
                else:
                    msg = msg + '%.3f' % value
            msg = msg + ', layer %d' % top_image
        self._statusBar().showMessage(msg)
