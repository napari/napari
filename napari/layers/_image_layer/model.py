from warnings import warn
from xml.etree.ElementTree import Element
from base64 import b64encode
from imageio import imwrite

import numpy as np
from copy import copy

import vispy.color

from .._base_layer import Layer
from ..._vispy.scene.visuals import Image as ImageNode

from ...util import is_multichannel
from ...util.interpolation import (interpolation_names,
                                  interpolation_index_to_name as _index_to_name,  # noqa
                                  interpolation_name_to_index as _name_to_index)  # noqa
from ...util.misc import guess_metadata
from ...util.colormaps import matplotlib_colormaps
from ...util.colormaps.vendored import cm
from ...util.event import Event

from .._register import add_to_viewer

from .view import QtImageLayer
from .view import QtImageControls


def _increment_unnamed_colormap(name, names):
    if name == '[unnamed colormap]':
        past_names = [n for n in names
                      if n.startswith('[unnamed colormap')]
        name = f'[unnamed colormap {len(past_names)}]'
    return name


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
    vispy_cmaps = vispy.color.get_colormaps()
    if name in vispy_cmaps:
        cmap = vispy.color.get_colormap(name)
    else:
        try:
            mpl_cmap = getattr(cm, name)
        except AttributeError:
            raise KeyError(f'Colormap "{name}" not found in either vispy '
                           'or matplotlib.')
        mpl_colors = mpl_cmap(np.linspace(0, 1, 256))
        cmap = vispy.color.Colormap(mpl_colors)
    return cmap


AVAILABLE_COLORMAPS = {k: vispy_or_mpl_colormap(k)
                       for k in matplotlib_colormaps +
                       list(vispy.color.get_colormaps())}


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
    name : str, keyword-only
        Name of the layer.
    clim_range : list | array | None
        Length two list or array with the default color limit range for the
        image. If not passed will be calculated as the min and max of the
        image. Passing a value prevents this calculation which can be useful
        when working with very large datasets that are dynamically loaded.
    **kwargs : dict
        Parameters that will be translated to metadata.
    """
    _colormaps = AVAILABLE_COLORMAPS

    default_colormap = 'magma'
    default_interpolation = 'nearest'

    def __init__(self, image, meta=None, multichannel=None, *, name=None,
                 clim_range=None, **kwargs):
        if name is None and meta is not None:
            if 'name' in meta:
                name = meta['name']

        visual = ImageNode(None, method='auto')
        super().__init__(visual, name)

        self.events.add(clim=Event,
                        colormap=Event,
                        interpolation=Event)

        meta = guess_metadata(image, meta, multichannel, kwargs)

        self._image = image
        self._meta = meta
        self.colormap_name = Image.default_colormap
        self.colormap = Image.default_colormap
        self.interpolation = Image.default_interpolation
        self._interpolation_names = interpolation_names

        # update flags
        self._need_display_update = False
        self._need_visual_update = False

        if clim_range is None:
            self._clim_range = self._clim_range_default()
        else:
            self._clim_range = clim_range
        self._node.clim = self._clim_range

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

    def _slice_image(self):
        """Determines the slice of image from the indices."""

        indices = list(self.indices)

        for dim in range(len(indices)):
            max_dim_index = self.image.shape[dim] - 1

            try:
                if indices[dim] > max_dim_index:
                    indices[dim] = max_dim_index
            except TypeError:
                pass

        self._image_view = np.asarray(self.image[tuple(indices)])

        return self._image_view

    def _set_view_slice(self):
        """Sets the view given the indices to slice with."""
        sliced_image = self._slice_image()

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
        return self.colormap_name, self._node.cmap

    @colormap.setter
    def colormap(self, colormap):
        name = '[unnamed colormap]'
        if isinstance(colormap, str):
            name = colormap
        elif isinstance(colormap, tuple):
            name, cmap = colormap
            self._colormaps[name] = cmap
        elif isinstance(colormap, dict):
            self._colormaps.update(colormap)
            name = list(colormap)[0]  # first key in dict
        elif isinstance(colormap, vispy.color.Colormap):
            name = _increment_unnamed_colormap(name,
                                               list(self._colormaps.keys()))
            self._colormaps[name] = colormap
        else:
            warn(f'invalid value for colormap: {colormap}')
            name = self.colormap_name
        self.colormap_name = name
        self._node.cmap = self._colormaps[name]
        self.events.colormap()

    @property
    def colormaps(self):
        """tuple of str: names of available colormaps.
        """
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
        self._clim_msg = f'{clim[0]: 0.3}, {clim[1]: 0.3}'
        self.status = self._clim_msg
        self._node.clim = clim
        self.events.clim()

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
        self.events.interpolation()

    @property
    def interpolation_functions(self):
        return tuple(interpolation_names)

    def _clim_range_default(self):
        return [float(self.image.min()), float(self.image.max())]

    def get_value(self):
        """Returns coordinates, values, and a string for a given mouse position
        and set of indices.

        Returns
        ----------
        coord : tuple of int
            Position of cursor in image space.
        value : int, float, or sequence of int or float
            Value of the data at the coord.
        """
        coord = np.round(self.coordinates).astype(int)
        coord[-2:] = np.clip(coord[-2:], 0,
                             np.asarray(self._image_view.shape) - 1)

        value = self._image_view[tuple(coord[-2:])]

        return coord, value

    def get_message(self, coord, value):
        """Generate a status message based on the coordinates and information
        about what shapes are hovered over

        Parameters
        ----------
        coord : sequence of int
            Position of mouse cursor in image coordinates.
        value : int or float or sequence of int or float
            Value of the data at the coord.

        Returns
        ----------
        msg : string
            String containing a message that can be used as a status update.
        """

        msg = f'{coord}, {self.name}' + ', value '
        if isinstance(value, np.ndarray):
            if isinstance(value[0], np.integer):
                msg = msg + str(value)
            else:
                v_str = '[' + str.join(', ', [f'{v:0.3}' for v in value]) + ']'
                msg = msg + v_str
        else:
            if isinstance(value, np.integer):
                msg = msg + str(value)
            else:
                msg = msg + f'{value:0.3}'

        return msg

    def to_xml_list(self):
        """Generates a list with a single xml element that defines the
        currently viewed image as a png according to the svg specification.

        Returns
        ----------
        xml : list of xml.etree.ElementTree.Element
            List of a single xml element specifying the currently viewed image
            as a png according to the svg specification.
        """
        image = np.clip(self._image_view, self.clim[0], self.clim[1])
        color_range = self.clim[1] - self.clim[0]
        if color_range != 0:
            image = image/color_range
        mapped_image = (self.colormap[1].map(image)*255).astype('uint8')
        mapped_image = mapped_image.reshape(list(self._image_view.shape) + [4])
        image_str = imwrite('<bytes>', mapped_image, format='png')
        image_str = "data:image/png;base64," + str(b64encode(image_str))[2:-1]
        props = {'xlink:href': image_str}
        width = str(self.shape[-1])
        height = str(self.shape[-2])
        opacity = str(self.opacity)
        xml = Element('image', width=width, height=height, opacity=opacity,
                      **props)
        return [xml]

    def on_mouse_move(self, event):
        """Called whenever mouse moves over canvas. Converts the `event.pos`
        from canvas coordinates to `self.coordinates` in image coordinates.
        """
        if event.pos is None:
            return
        self.coordinates = event.pos
        coord, value = self.get_value()
        self.status = self.get_message(coord, value)
