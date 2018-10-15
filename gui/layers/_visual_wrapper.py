from vispy.visuals.transforms import ChainTransform, NullTransform, STTransform


class VisualWrapper:
    def __init__(self, central_node):
        self._node = central_node

        # TODO: create & use our own transformation classes
        vt = self._node.transforms.visual_transform
        vt.transforms = [STTransform(),
                         ChainTransform(NullTransform())]

    @property
    def _master_transform(self):
        """vispy.visuals.transforms.STTransform:
        Central node's firstmost transform.
        """
        return self._node.transforms.visual_transform.transforms[0]

    @property
    def _transforms(self):
        """tuple of vispy.visuals.transforms.BaseTransform:
        Transforms to apply to the central node.
        """
        tr = self._node.transforms.visual_transform.transforms[1]
        return tuple(tr.transforms)

    @_transforms.setter
    def _transforms(self, transforms):
        if tuple(transforms) == self._transforms:
            return
        tr = self._node.transforms.visual_transform.transforms[1]
        tr.transforms = list(transforms)

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
