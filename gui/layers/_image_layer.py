import weakref

import numpy as np

from ._base_layer import Layer
from .._vispy.scene.visuals import Image as ImageNode

from ..util import is_multichannel
from ..util.misc import guess_metadata
from ..util.interpolation import (interpolation_names,
                                  interpolation_index_to_name as _index_to_name,  # noqa
                                  interpolation_name_to_index as _name_to_index)  # noqa

from vispy.color.colormap import get_colormaps

from ._register import add_to_viewer

from ..elements.qt import QtImageLayer

@add_to_viewer
class Image(Layer):
    """Image layer.

    Parameters
    ----------
    image : np.ndarray
        Image data.
    meta : dict
        Image metadata.
    """
    _colormaps = get_colormaps()

    default_cmap = 'hot'
    default_interpolation = 'nearest'

    def __init__(self, image, meta):
        visual = ImageNode(None, method='auto')
        super().__init__(visual)

        self._image = image
        self._meta = meta
        self.cmap = Image.default_cmap
        self.interpolation = Image.default_interpolation

        # update flags
        self._need_display_update = False
        self._need_visual_update = False

        self._qt = QtImageLayer(self)

    @property
    def image(self):
        """np.ndarray: Image data.
        """
        return self._image

    @image.setter
    def image(self, image):
        self._image = image

        self.refresh()

    @property
    def meta(self):
        """dict: Image metadata.
        """
        return self._meta

    @meta.setter
    def meta(self, meta):
        self._meta = meta

        self.refresh()

    @property
    def data(self):
        """tuple of np.ndarray, dict: Image data and metadata.
        """
        return self.image, self.meta

    @data.setter
    def data(self, data):
        self._image, self._meta = data
        self.refresh()

    def _get_shape(self):
        if self.multichannel:
            return self.image.shape[:-1]
        return self.image.shape

    def _update(self):
        """Update the underlying visual.
        """
        if self._need_display_update:
            self._need_display_update = False

            self.viewer._child_layer_changed = True
            self.viewer._update()

            self._node._need_colortransform_update = True
            self._set_view_slice(self.viewer.indices)

        if self._need_visual_update:
            self._need_visual_update = False
            self._node.update()

    def _refresh(self):
        """Fully refresh the underlying visual.
        """
        self._need_display_update = True
        self._update()

    def _set_view_slice(self, indices):
        """Sets the view given the indices to slice with.

        Parameters
        ----------
        indices : sequence of int or slice
            Indices to slice with.
        """
        ndim = self.ndim

        indices = list(indices)
        indices = indices[:ndim]

        for dim in range(len(indices)):
            max_dim_index = self.image.shape[dim] - 1

            try:
                if indices[dim] > max_dim_index:
                    indices[dim] = max_dim_index
            except TypeError:
                pass

        sliced_image = self.image[tuple(indices)]

        self._node.set_data(sliced_image)

        self._need_visual_update = True
        self._update()

    @property
    def multichannel(self):
        """bool: Whether the image is multichannel.
        """
        return is_multichannel(self.meta)

    @multichannel.setter
    def multichannel(self, val):
        if val == self.multichannel:
            return

        self.meta['itype'] = 'multi'

        self._need_display_update = True
        self._update()

    @property
    def interpolation_index(self):
        """int: Index of the current interpolation method equipped.
        """
        return self._interpolation_index

    @interpolation_index.setter
    def interpolation_index(self, interpolation_index):
        intp_index = interpolation_index % len(interpolation_names)
        self._interpolation_index = intp_index
        self._node.interpolation = _index_to_name(intp_index)

    @property
    def colormap(self):
        """string or ColorMap: Colormap to use for luminance images.
        """
        return self.cmap

    @colormap.setter
    def colormap(self, colormap):
        self.cmap = colormap

    @property
    def colormaps(self):
        """tuple of str: Colormap names.
        """
        # TODO: achieve this by wrapping DictKeys
        return tuple(self._colormaps.keys())

    # wrap visual properties:

    @property
    def clim(self):
        """string or tuple of float: Limits to use for the colormap.
        Can be 'auto' to auto-set bounds to the min and max of the data.
        """
        return self._node.clim

    @clim.setter
    def clim(self, clim):
        self._node.clim = clim

    @property
    def cmap(self):
        """string or ColorMap: Colormap to use for luminance images.
        """
        return self._node.cmap

        for name, obj in Image._colormaps.items():
            if obj == cmap:
                return name
        else:
            return cmap

    @cmap.setter
    def cmap(self, cmap):
        try:
            cmap = Image._colormaps[cmap]
        except KeyError:
            pass

        self._node.cmap = cmap

    @property
    def method(self):
        """string: Selects method of rendering image in case of non-linear
        transforms. Each method produces similar results, but may trade
        efficiency and accuracy. If the transform is linear, this parameter
        is ignored and a single quad is drawn around the area of the image.

            * 'auto': Automatically select 'impostor' if the image is drawn
              with a nonlinear transform; otherwise select 'subdivide'.
            * 'subdivide': ImageVisual is represented as a grid of triangles
              with texture coordinates linearly mapped.
            * 'impostor': ImageVisual is represented as a quad covering the
              entire view, with texture coordinates determined by the
              transform. This produces the best transformation results, but may
              be slow.
        """
        return self._node.method

    @method.setter
    def method(self):
        self._node.method = method

    @property
    def interpolation(self):
        """string: Equipped interpolation method's name.
        """
        return _index_to_name(self.interpolation_index)

    @interpolation.setter
    def interpolation(self, interpolation):
        self.interpolation_index = _name_to_index(interpolation)

    @property
    def interpolation_functions(self):
        return tuple(interpolation_names)
