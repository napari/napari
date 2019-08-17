from vispy.visuals.transforms import STTransform
from contextlib import contextmanager
from ..util.event import EmitterGroup, Event
from abc import ABC, abstractmethod


class VispyBaseLayer(ABC):
    """Wrapper around ``vispy.scene.VisualNode`` objects.

    Meant to be subclassed.

    Parameters
    ----------
    central_node : vispy.scene.VisualNode
        Central node/control point with which to interact with the visual.
        Stored as ``_node``.

    Attributes
    ----------
    opacity : flaot
        Opacity of the layer visual, between 0.0 and 1.0.
    visible : bool
        Whether the layer visual is currently being displayed.
    blending : Blending
        Determines how RGB and alpha values get mixed.
            Blending.OPAQUE
                Allows for only the top layer to be visible and corresponds to
                depth_test=True, cull_face=False, blend=False.
            Blending.TRANSLUCENT
                Allows for multiple layers to be blended with different opacity
                and corresponds to depth_test=True, cull_face=False,
                blend=True, blend_func=('src_alpha', 'one_minus_src_alpha').
            Blending.ADDITIVE
                Allows for multiple layers to be blended together with
                different colors and opacity. Useful for creating overlays. It
                corresponds to depth_test=False, cull_face=False, blend=True,
                blend_func=('src_alpha', 'one').
    scale : sequence of float
        Scale factors for the layer visual in the scenecanvas.
    translate : sequence of float
        Translation values for the layer visual in the scenecanvas.
    z_index : int
        Depth of the layer visual relative to other visuals in the scenecanvas.

    Extended Summary
    ----------
    _master_transform : vispy.visuals.transforms.STTransform
        Transform positioning the layer visual inside the scenecanvas.
    _order : int
        Order in which the visual is drawn in the scenegraph. Lower values
        are closer to the viewer.
    _parent : vispy.View
        View containing parent node and camera.
    """

    def __init__(self, layer, node):
        super().__init__()

        self.layer = layer
        self.node = node

        self.layer.events.refresh.connect(lambda e: self.node.update())
        self.layer.events.set_data.connect(lambda e: self._on_data_change())
        self.layer.dims.events.axis.connect(lambda e: self._update_coordinates())

        self.layer.events.visible.connect(lambda e: self._on_visible_change())
        self.layer.events.opacity.connect(lambda e: self._on_opacity_change())

        self._on_visible_change()
        self._on_opacity_change()
        self._on_blending_change()

    @abstractmethod
    def _on_data_change(self):
        raise NotImplementedError()

    def _on_visible_change(self):
        self.node.visible = self.layer.visible

    def _on_opacity_change(self):
        self.node.visible = self.layer.opacity

    def _on_blending_change(self):
        self.node.set_gl_state(self.layer.blending)
        self.node.update()

    #     self._update_properties = False
    #
    # @contextmanager
    # def block_update_properties(self):
    #     self._update_properties = False
    #     yield
    #     self._update_properties = True
    #
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

    #
    # @blending.setter
    # def blending(self, blending):
    #     if isinstance(blending, str):
    #         blending = Blending(blending)
    #
    #     self._node.set_gl_state(blending.value)
    #     self._blending = blending
    #     self._node.update()
    #     self.events.blending()
    #
    #
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

    # @property
    # def scale_factor(self):
    #     """float: Conversion factor from canvas coordinates to image
    #     coordinates, which depends on the current zoom level.
    #     """
    #     if self._node.canvas is not None:
    #         transform = self._node.canvas.scene.node_transform(self._node)
    #         scale_factor = transform.map([1, 1])[0] - transform.map([0, 0])[0]
    #     else:
    #         scale_factor = 1
    #     return scale_factor
    #

    # def _update(self):
    #     """Update the underlying visual."""
    #     if self._need_display_update:
    #         self._need_display_update = False
    #         if hasattr(self._node, '_need_colortransform_update'):
    #             self._node._need_colortransform_update = True
    #         self._set_view_slice()
    #
    #     if self._need_visual_update:
    #         self._need_visual_update = False
    #         self.node.update()

    def _update_coordinates(self):
        """Insert the cursor position (x, y) into the correct position in the
        tuple of indices and update the cursor coordinates.
        """
        if self.canvas is not None:
            transform = self.canvas.scene.node_transform(self)
            position = transform.map(list(self.layer.position))[
                : len(self.layer.dims.displayed)
            ]
            position = position[::-1]
        else:
            position = [0] * len(self.layer.dims.displayed)

        coords = list(self.layer.dims.indices)
        for d, p in zip(self.layer.dims.displayed, position):
            coords[d] = p
        self.layer.coordinates = tuple(coords)

    def on_mouse_move(self, event):
        """Called whenever mouse moves over canvas.
        """
        return

    def on_mouse_press(self, event):
        """Called whenever mouse pressed in canvas.
        """
        return

    def on_mouse_release(self, event):
        """Called whenever mouse released in canvas.
        """
        return

    def on_draw(self, event):
        """Called whenever the canvas is drawn.
        """
        return
