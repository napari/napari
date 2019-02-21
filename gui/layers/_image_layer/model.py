import numpy as np
from numpy import clip, integer, ndarray, append, insert, delete, empty
from copy import copy
from ..vendored import cm

from .._base_layer import Layer
from ..._vispy.scene.visuals import Image as ImageNode

from ...util import is_multichannel
from ...util.interpolation import (interpolation_names,
                                  interpolation_index_to_name as _index_to_name,  # noqa
                                  interpolation_name_to_index as _name_to_index)  # noqa
from ...util.misc import guess_metadata

from vispy import color
from ..colormaps import matplotlib_colormaps

from .._register import add_to_viewer

from .view import QtImageLayer
from .view import QtImageControls


def vispy_or_mpl_colormap(name):
    """Try to get a colormap from vispy, or convert an mpl one to vispy format.

    Parameters
    ----------
    name : str
        The name of the colormap.

    Returns
    -------
    cmap : vispy.color.Colormap
        The found colormap.

    Raises
    ------
    KeyError
        If no colormap with that name is found within vispy or matplotlib.
    """
    vispy_cmaps = color.get_colormaps()
    if name in vispy_cmaps:
        cmap = color.get_colormap(name)
    else:
        try:
            mpl_cmap = getattr(cm, name)
        except AttributeError:
            raise KeyError(f'Colormap "{name}" not found in either vispy '
                           'or matplotlib.')
        mpl_colors = mpl_cmap(np.linspace(0, 1, 256))
        cmap = color.Colormap(mpl_colors)
    return cmap


AVAILABLE_COLORMAPS = {k: vispy_or_mpl_colormap(k)
                       for k in matplotlib_colormaps +
                       list(color.get_colormaps())}


@add_to_viewer
class Image(Layer):
    """Image layer.

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
    """
    _colormaps = AVAILABLE_COLORMAPS

    default_cmap = 'magma'
    default_interpolation = 'nearest'

    def __init__(self, image, meta=None, multichannel=None, **kwargs):
        self.visual = ImageNode(None, method='auto')
        super().__init__(self.visual)

        meta = guess_metadata(image, meta, multichannel, kwargs)

        self._image = image
        self._meta = meta
        self.cmap = Image.default_cmap
        self.interpolation = Image.default_interpolation
        self._interpolation_names = interpolation_names

        # update flags
        self._need_display_update = False
        self._need_visual_update = False

        self.name = 'image'

        self._clim_range = self._clim_range_default()
        self._node.clim = [np.min(self.image), np.max(self.image)]

        cmin, cmax = self.clim
        self._clim_msg = f'{cmin: 0.3}, {cmax: 0.3}'

        self._qt_properties = QtImageLayer(self)
        self._qt_controls = QtImageControls(self)

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

            self.viewer.dimensions._child_layer_changed = True
            self.viewer.dimensions._update()

            self._node._need_colortransform_update = True
            self._set_view_slice(self.viewer.dimensions.indices)

        if self._need_visual_update:
            self._need_visual_update = False
            self._node.update()

    def _refresh(self):
        """Fully refresh the underlying visual.
        """
        self._need_display_update = True
        self._update()

    def _slice_image(self, indices):
        """Determines the slice of image given the indices.

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

        return self.image[tuple(indices)]

    def _set_view_slice(self, indices):
        """Sets the view given the indices to slice with.

        Parameters
        ----------
        indices : sequence of int or slice
            Indices to slice with.
        """
        sliced_image = self._slice_image(indices)

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
        """tuple of str: names of available colormaps.
        """
        return tuple(AVAILABLE_COLORMAPS.keys())

    # wrap visual properties:
    @property
    def clim(self):
        """string or tuple of float: Limits to use for the colormap.
        Can be 'auto' to auto-set bounds to the min and max of the data.
        """
        return self._node.clim

    @clim.setter
    def clim(self, clim):
        self._clim_msg = f'{clim[0]: 0.3}, {clim[1]: 0.3}'
        self.status = self._clim_msg
        self._node.clim = clim

    @property
    def cmap(self):
        """string or ColorMap: Colormap to use for luminance images.
        """
        return self._node.cmap

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
    def method(self, method):
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

    def _clim_range_default(self):
        return [np.min(self.image), np.max(self.image)]

    def get_value(self, position, indices):
        """Returns coordinates, values, and a string for a given mouse position
        and set of indices.

        Parameters
        ----------
        position : sequence of two int
            Position of mouse cursor in canvas.
        indices : sequence of int or slice
            Indices that make up the slice.

        Returns
        ----------
        coord : sequence of int
            Position of mouse cursor in data.
        value : int or float or sequence of int or float
            Value of the data at the coord.
        msg : string
            String containing a message that can be used as
            a status update.
        """
        transform = self._node.canvas.scene.node_transform(self._node)
        pos = transform.map(position)
        pos = [clip(pos[1], 0, self.shape[0]-1), clip(pos[0], 0,
                                                      self.shape[1]-1)]
        coord = copy(indices)
        coord[0] = int(pos[0])
        coord[1] = int(pos[1])
        value = self._slice_image(coord)
        msg = f'{coord}, {self.name}' + ', value '
        if isinstance(value, ndarray):
            if isinstance(value[0], integer):
                msg = msg + str(value)
            else:
                v_str = '[' + str.join(', ', [f'{v:0.3}' for v in value]) + ']'
                msg = msg + v_str
        else:
            if isinstance(value, integer):
                msg = msg + str(value)
            else:
                msg = msg + f'{value:0.3}'
        return coord, value, msg

    def on_mouse_move(self, event):
        """Called whenever mouse moves over canvas.
        """
        if event.pos is None:
            return
        coord, value, msg = self.get_value(event.pos,
                                           self.viewer.dimensions.indices)
        self.status = msg
