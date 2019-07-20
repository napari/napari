from vispy.visuals.transforms import STTransform
from contextlib import contextmanager
from ...util.event import EmitterGroup, Event

from ._constants import Blending


class VisualWrapper:
    """Wrapper around ``vispy.scene.VisualNode`` objects.

    Meant to be subclassed.

    Parameters
    ----------
    central_node : vispy.scene.VisualNode
        Central node/control point with which to interact with the visual.
        Stored as ``_node``.
    opacity : float
        Opacity of the layer visual, between 0.0 and 1.0.
    blending : str
        One of a list of preset blending modes that determines how RGB and
        alpha values of the layer visual get mixed. Allowed values are
        {'opaque', 'translucent', and 'additive'}.
    visible : bool
        Whether the layer visual is currently being displayed.

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

    def __init__(
        self, central_node, *, opacity=1, blending='translucent', visible=True
    ):
        super().__init__()
        self._node = central_node
        self._parent = None
        self.events = EmitterGroup(
            source=self,
            auto_connect=True,
            blending=Event,
            opacity=Event,
            visible=Event,
        )
        self._update_properties = False
        self.opacity = opacity
        self.blending = blending
        self.visible = visible

    @contextmanager
    def block_update_properties(self):
        self._update_properties = False
        yield
        self._update_properties = True

    @property
    def _master_transform(self):
        """vispy.visuals.transforms.STTransform:
        Central node's firstmost transform.
        """
        # whenever a new parent is set, the transform is reset
        # to a NullTransform so we reset it here
        if not isinstance(self._node.transform, STTransform):
            self._node.transform = STTransform()

        return self._node.transform

    @property
    def _order(self):
        """int: Order in which the visual is drawn in the scenegraph.

        Lower values are closer to the viewer.
        """
        return self._node.order

    @_order.setter
    def _order(self, order):
        # workaround for opacity (see: #22)
        order = -order
        self.z_index = order
        # end workaround
        self._node.order = order

    @property
    def parent(self):
        """vispy.View: View containing parent node and camera.
        """
        return self._parent

    @parent.setter
    def parent(self, parent):
        self._parent = parent
        self._node.parent = parent.scene

    @property
    def opacity(self):
        """float: Opacity value between 0.0 and 1.0.
        """
        return self._node.opacity

    @opacity.setter
    def opacity(self, opacity):
        if not 0.0 <= opacity <= 1.0:
            raise ValueError(
                'opacity must be between 0.0 and 1.0; ' f'got {opacity}'
            )

        self._node.opacity = opacity
        self.events.opacity()

    @property
    def blending(self):
        """Blending mode: Determines how RGB and alpha values get mixed.

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
        """
        return str(self._blending)

    @blending.setter
    def blending(self, blending):
        if isinstance(blending, str):
            blending = Blending(blending)

        self._node.set_gl_state(blending.value)
        self._blending = blending
        self._node.update()
        self.events.blending()

    @property
    def visible(self):
        """bool: Whether the visual is currently being displayed."""
        return self._node.visible

    @visible.setter
    def visible(self, visibility):
        self._node.visible = visibility
        self.events.visible()

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
