from numpy import clip, integer, ndarray, append, insert, delete, empty
from copy import copy

from vispy.util.event import EmitterGroup, Event

from .view import QtViewer


class Viewer:
    """Viewer containing the rendered scene, layers, and controlling elements
    including dimension sliders, and control bars for color limits.

    Attributes
    ----------
    window : Window
        Parent window.
    layers : LayersList
        List of contained layers.
    dimensions : Dimensions
        Contains axes, indices, dimensions and sliders.
    controls : Controls
        Contains layer specific controls.
    camera : vispy.scene.Camera
        Viewer camera.
    """

    def __init__(self):
        super().__init__()
        from .._layers_list import LayersList
        from .._controls import Controls
        from .._dimensions import Dimensions

        self.events = EmitterGroup(source=self,
                                   auto_connect=True,
                                   status=Event,
                                   help=Event,
                                   active_markers=Event)
        self.dimensions = Dimensions(self)
        self.layers = LayersList(self)
        self.controls = Controls(self)

        self._active_image = None
        self._active_markers = None
        self._visible_markers = []
        self.position = [0, 0]

        self._status = 'Ready'
        self._help = ''

        self._qt = QtViewer(self)

    @property
    def _canvas(self):
        return self._qt.canvas

    @property
    def _view(self):
        return self._qt.view

    @property
    def camera(self):
        """vispy.scene.Camera: Viewer camera.
        """
        return self._view.camera

    @property
    def status(self):
        """string: Status string
        """
        return self._status

    @status.setter
    def status(self, status):
        if status == self.status:
            return
        self._status = status
        self.events.status(text=self._status)

    @property
    def help(self):
        """string: String that can be displayed to the
        user in the status bar with helpful usage tips.
        """
        return self._help

    @help.setter
    def help(self, help):
        if help == self.help:
            return
        self._help = help
        self.events.help(text=self._help)

    @property
    def active_markers(self):
        """int: index of active_markers
        """
        return self._active_markers

    @active_markers.setter
    def active_markers(self, active_markers):
        if active_markers == self.active_markers:
            return
        self._active_markers = active_markers
        self.events.active_markers(index=self._active_markers)

    def reset_view(self):
        """Resets the camera's view.
        """
        self.camera.set_range()

    def screenshot(self, region=None, size=None, bgcolor=None, crop=None):
        """Render the scene to an offscreen buffer and return the image array.

        Parameters
        ----------
        region : tuple | None
            Specifies the region of the canvas to render. Format is
            (x, y, w, h). By default, the entire canvas is rendered.
        size : tuple | None
            Specifies the size of the image array to return. If no size is
            given, then the size of the *region* is used, multiplied by the
            pixel scaling factor of the canvas (see `pixel_scale`). This
            argument allows the scene to be rendered at resolutions different
            from the native canvas resolution.
        bgcolor : instance of Color | None
            The background color to use.
        crop : array-like | None
            If specified it determines the pixels read from the framebuffer.
            In the format (x, y, w, h), relative to the region being rendered.
        Returns
        -------
        image : array
            Numpy array of type ubyte and shape (h, w, 4). Index [0, 0] is the
            upper-left corner of the rendered region.

        """
        return self._canvas.render(region=None, size=None, bgcolor=None,
                                   crop=None)

    def add_layer(self, layer):
        """Adds a layer to the viewer.

        Parameters
        ----------
        layer : Layer
            Layer to add.
        """
        self.layers.append(layer)
        if len(self.layers) == 1:
            self.reset_view()

    def _new_markers(self):
        if self.dimensions.max_dims == 0:
            empty_markers = empty((0, 2))
        else:
            empty_markers = empty((0, self.dimensions.max_dims))
        self.add_markers(empty_markers)

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
        self._update_status()

    def _update_active_layers(self, event):
        from ...layers._image_layer import Image
        from ...layers._markers_layer import Markers
        top_markers = []
        for i, layer in enumerate(self.layers[::-1]):
            if layer.visible and isinstance(layer, Image):
                top_image = len(self.layers) - 1 - i
                break
            elif layer.visible and isinstance(layer, Markers):
                top_markers.append(len(self.layers) - 1 - i)
        else:
            top_image = None

        active_markers = None
        for i in top_markers:
            if self.layers[i].selected:
                active_markers = i
                break

        self._active_image = top_image
        self._visible_markers = top_markers
        self.active_markers = active_markers
        self.controls.update()

    def _update_status(self):
        msg = ''
        for i in self._visible_markers:
            layer = self.layers[i]
            coord, value, msg = layer.get_value(self.position,
                                                self.dimensions.indices)
            if value is None:
                pass
            else:
                break
        else:
            if self._active_image is None:
                pass
            else:
                layer = self.layers[self._active_image]
                coord, value, msg = layer.get_value(self.position,
                                                    self.dimensions.indices)
        self.status = msg
