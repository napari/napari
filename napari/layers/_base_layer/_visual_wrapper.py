# TODO: create & use our own transform class
from vispy.visuals.transforms import STTransform
from ...util.event import EmitterGroup, Event

from ._constants import Blending


class VisualWrapper:
    """Wrapper around ``vispy.scene.VisualNode`` objects.
    Meant to be subclassed.

    "Hidden" properties:
        * ``_master_transform``
        * ``_order``
        * ``_parent``

    Parameters
    ----------
    central_node : vispy.scene.VisualNode
        Central node/control point with which to interact with the visual.
        Stored as ``_node``.

    Attributes
    ----------
    opacity
    visible
    scale
    blending
    translate
    z_index

    Notes
    -----
    It is recommended to use the backported ``vispy`` nodes
    at ``_vispy.scene.visuals`` for various bug fixes.
    """

    def __init__(self, central_node):
        self._node = central_node
        self._blending = Blending.TRANSLUCENT
        self._parent = None
        self.events = EmitterGroup(
            source=self,
            auto_connect=True,
            blending=Event,
            opacity=Event,
            visible=Event,
        )

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
        """Blending: Blending mode.
            Selects a preset blending mode in vispy that determines how
            RGB and alpha values get mixed.
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
        """bool: Whether the visual is currently being displayed.
        """
        return self._node.visible

    @visible.setter
    def visible(self, visibility):
        self._node.visible = visibility
        self.events.visible()

    @property
    def scale(self):
        """sequence of float: Scale factors.
        """
        return self._master_transform.scale

    @scale.setter
    def scale(self, scale):
        self._master_transform.scale = scale

    @property
    def translate(self):
        """sequence of float: Translation values.
        """
        return self._master_transform.translate

    @translate.setter
    def translate(self, translate):
        self._master_transform.translate = translate

    @property
    def z_index(self):
        return -self._master_transform.translate[2]

    @z_index.setter
    def z_index(self, index):
        tr = self._master_transform
        tl = tr.translate
        tl[2] = -index

        tr.translate = tl
