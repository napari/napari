# TODO: create & use our own transform class
from vispy.visuals.transforms import STTransform


class VisualWrapper:
    def __init__(self, central_node):
        self._node = central_node

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
    def _parent(self):
        """vispy.scene.Node: Parent node.
        """
        return self._node.parent

    @_parent.setter
    def _parent(self, parent):
        self._node.parent = parent

    @property
    def opacity(self):
        """float: Opacity value between 0.0 and 1.0.
        """
        return self._node.opacity

    @opacity.setter
    def opacity(self, opacity):
        if not 0.0 <= opacity <= 1.0:
            raise ValueError('opacity must be between 0.0 and 1.0; '
                             f'got {opacity}')

        self._node.opacity = opacity

    @property
    def visible(self):
        """bool: Whether the visual is currently being displayed.
        """
        return self._node.visible

    @visible.setter
    def visible(self, visibility):
        self._node.visible = visibility

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
