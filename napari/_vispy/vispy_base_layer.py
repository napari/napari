from vispy.visuals.transforms import STTransform
from contextlib import contextmanager
from ..util.event import EmitterGroup, Event
from ..util.layer_view import LayerView
from abc import ABC, abstractmethod


class VispyBaseLayer(LayerView, ABC):
    """Base object for individual layer views

    Meant to be subclassed.

    Parameters
    ----------
    layer : napari.layers.Layer
        Layer model.
    node : vispy.scene.VisualNode
        Central node with which to interact with the visual.

    Attributes
    ----------
    layer : napari.layers.Layer
        Layer model.
    node : vispy.scene.VisualNode
        Central node with which to interact with the visual.
    scale : sequence of float
        Scale factors for the layer visual in the scenecanvas.
    translate : sequence of float
        Translation values for the layer visual in the scenecanvas.
    z_index : int
        Depth of the layer visual relative to other visuals in the scenecanvas.
    scale_factor : float
        Conversion factor from canvas coordinates to image coordinates, which
        depends on the current zoom level.

    Extended Summary
    ----------
    _master_transform : vispy.visuals.transforms.STTransform
        Transform positioning the layer visual inside the scenecanvas.
    _order : int
        Order in which the visual is drawn in the scenegraph. Lower values
        are closer to the viewer.
    """

    def __init__(self, layer, node):
        super().__init__()

        self.layer = layer
        self.node = node
        self._position = (0,) * self.layer.ndim
        self.camera = None

        self.layer.events.refresh.connect(lambda e: self.node.update())
        self.layer.events.set_data.connect(lambda e: self._on_data_change())

        self.layer.events.visible.connect(lambda e: self._on_visible_change())
        self.layer.events.opacity.connect(lambda e: self._on_opacity_change())
        self.layer.events.blending.connect(
            lambda e: self._on_blending_change()
        )
        self.layer.events.scale.connect(lambda e: self._on_scale_change())
        self.layer.events.translate.connect(
            lambda e: self._on_translate_change()
        )

        self._on_visible_change()
        self._on_opacity_change()
        self._on_blending_change()
        self._on_scale_change()
        self._on_translate_change()

    @property
    def _master_transform(self):
        """vispy.visuals.transforms.STTransform:
        Central node's firstmost transform.
        """
        # whenever a new parent is set, the transform is reset
        # to a NullTransform so we reset it here
        if not isinstance(self.node.transform, STTransform):
            self.node.transform = STTransform()

        return self.node.transform

    @property
    def _order(self):
        """int: Order in which the visual is drawn in the scenegraph.

        Lower values are closer to the viewer.
        """
        return self.node.order

    @_order.setter
    def _order(self, order):
        # workaround for opacity (see: #22)
        order = -order
        self.z_index = order
        # end workaround
        self.node.order = order

    @property
    def scale(self):
        """sequence of float: Scale factors."""
        return self._master_transform.scale

    @scale.setter
    def scale(self, scale):
        self._master_transform.scale = scale

    @property
    def translate(self):
        """sequence of float: Translation values."""
        return self._master_transform.translate

    @translate.setter
    def translate(self, translate):
        self._master_transform.translate = translate

    @property
    def z_index(self):
        """int: Depth of the visual in the scenecanvas."""
        return -self._master_transform.translate[2]

    @z_index.setter
    def z_index(self, index):
        tr = self._master_transform
        tl = tr.translate
        tl[2] = -index
        tr.translate = tl

    @property
    def scale_factor(self):
        """float: Conversion factor from canvas coordinates to image
        coordinates, which depends on the current zoom level.
        """
        transform = self.node.canvas.scene.node_transform(self.node)
        scale_factor = transform.map([1, 1])[0] - transform.map([0, 0])[0]
        return scale_factor

    @abstractmethod
    def _on_data_change(self):
        raise NotImplementedError()

    def _on_visible_change(self):
        self.node.visible = self.layer.visible

    def _on_opacity_change(self):
        self.node.opacity = self.layer.opacity

    def _on_blending_change(self):
        self.node.set_gl_state(self.layer.blending)
        self.node.update()

    def _on_scale_change(self):
        self.scale = [
            self.layer.scale[d] for d in self.layer.dims.displayed[::-1]
        ]
        self.layer.position = self._transform_position(self._position)

    def _on_translate_change(self):
        self.translate = [
            self.layer.translate[d] for d in self.layer.dims.displayed[::-1]
        ]
        self.layer.position = self._transform_position(self._position)

    def _transform_position(self, position):
        """Transform cursor position from canvas space (x, y) into image space.

        Parameters
        -------
        position : 2-tuple
            Cursor position in canvase (x, y).

        Returns
        -------
        coords : tuple
            Coordinates of cursor in image space for displayed dimensions only
        """
        if self.node.canvas is not None:
            transform = self.node.canvas.scene.node_transform(self.node)
            position = transform.map(list(position))[
                : len(self.layer.dims.displayed)
            ]
            coords = tuple(position[::-1])
        else:
            coords = (0,) * len(self.layer.dims.displayed)
        return coords

    def on_mouse_move(self, event):
        """Called whenever mouse moves over canvas."""
        if event.pos is None:
            return
        self._position = list(event.pos)
        self.layer.position = self._transform_position(self._position)
        self.layer.on_mouse_move(event)

    def on_mouse_press(self, event):
        """Called whenever mouse pressed in canvas.
        """
        if event.pos is None:
            return
        self._position = list(event.pos)
        self.layer.position = self._transform_position(self._position)
        self.layer.on_mouse_press(event)

    def on_mouse_release(self, event):
        """Called whenever mouse released in canvas.
        """
        if event.pos is None:
            return
        self._position = list(event.pos)
        self.layer.position = self._transform_position(self._position)
        self.layer.on_mouse_release(event)

    def on_draw(self, event):
        """Called whenever the canvas is drawn.
        """
        self.layer.scale_factor = self.scale_factor
