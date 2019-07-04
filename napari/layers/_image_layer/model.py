from warnings import warn
from xml.etree.ElementTree import Element
from base64 import b64encode
from imageio import imwrite

import numpy as np
from copy import copy
from scipy import ndimage as ndi

import vispy.color

from .._base_layer import Layer
from ..._vispy.scene.visuals import Image as ImageNode

# from ...util import is_multichannel

from ...util.misc import guess_multichannel
from ...util.colormaps import matplotlib_colormaps, simple_colormaps
from ...util.colormaps.vendored import cm
from ...util.event import Event
from ._constants import Interpolation


def _increment_unnamed_colormap(name, names):
    if name == '[unnamed colormap]':
        past_names = [n for n in names if n.startswith('[unnamed colormap')]
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
            raise KeyError(
                f'Colormap "{name}" not found in either vispy '
                'or matplotlib.'
            )
        mpl_colors = mpl_cmap(np.linspace(0, 1, 256))
        cmap = vispy.color.Colormap(mpl_colors)
    return cmap


# A dictionary mapping names to VisPy colormap objects
ALL_COLORMAPS = {k: vispy_or_mpl_colormap(k) for k in matplotlib_colormaps}
ALL_COLORMAPS.update(simple_colormaps)

# ... sorted alphabetically by name
AVAILABLE_COLORMAPS = {k: v for k, v in sorted(ALL_COLORMAPS.items())}


class Image(Layer):
    """Image layer.

    Parameters
    ----------
    image : np.ndarray
        Image data.
    metadata : dict, optional
        Image metadata.
    multichannel : bool, optional
        Whether the image is multichannel. Guesses if None.
    name : str, keyword-only
        Name of the layer.
    clim_range : list | array | None
        Length two list or array with the default color limit range for the
        image. If not passed will be calculated as the min and max of the
        image. Passing a value prevents this calculation which can be
        useful when working with very large datasets that are dynamically
        loaded.
    **kwargs : dict
        Parameters that will be translated to metadata.
    """

    _colormaps = AVAILABLE_COLORMAPS

    default_interpolation = str(Interpolation.NEAREST)

    def __init__(
        self,
        image,
        colormap='gray',
        clim=None,
        clim_range=None,
        multichannel=None,
        interpolation='nearest',
        metadata={},
        *,
        name=None,
        **kwargs,
    ):

        visual = ImageNode(None, method='auto')
        super().__init__(visual, name)

        self.events.add(clim=Event, colormap=Event, interpolation=Event)

        with self.freeze_refresh():
            self._data = image
            self.metadata = metadata
            self.multichannel = multichannel
            # Intitialize image views and thumbnails with zeros
            self._data_view = np.zeros(
                (1, 1) + (self.shape[-1],) * self.multichannel
            )
            self._data_thumbnail = self._data_view

            self._colormap_name = ''
            self._clim_msg = ''

            if clim_range is None:
                self._clim_range = self._clim_range_default()
            else:
                self._clim_range = clim_range

            if clim is None:
                self.clim = self._clim_range
            else:
                self.clim = clim

            self.colormap = colormap
            self.interpolation = interpolation

        # update flags
        self._need_display_update = False
        self._need_visual_update = False

    @property
    def data(self):
        """np.ndarray: Image data.
        """
        return self._data

    @data.setter
    def data(self, data):
        self._data = data
        if self.multichannel:
            self._multichannel = guess_multichannel(data.shape)
        self.events.data()
        self.refresh()

    def _get_shape(self):
        if self.multichannel:
            return self.data.shape[:-1]
        return self.data.shape

    def _slice_data(self):
        """Determines the slice of data from the indices."""

        indices = list(self.indices)
        indices[:-2] = np.clip(
            indices[:-2], 0, np.subtract(self.shape[:-2], 1)
        )
        self._data_view = np.asarray(self.data[tuple(indices)])
        self._data_thumbnail = self._data_view

        return self._data_view

    def _set_view_slice(self):
        """Sets the view given the indices to slice with."""
        sliced_data = self._slice_data()

        self._node.set_data(sliced_data)

        self._need_visual_update = True
        self._update()

        coord, value = self.get_value()
        self.status = self.get_message(coord, value)
        self._update_thumbnail()

    @property
    def multichannel(self):
        """bool: Whether the image is multichannel.
        """
        return self._multichannel

    @multichannel.setter
    def multichannel(self, multichannel):
        if multichannel is False:
            self._multichannel = multichannel
        else:
            # If multichannel is True or None then guess if multichannel
            # allowed or not, and if allowed set it to be true
            self._multichannel = guess_multichannel(self.data.shape)
        self.refresh()

    @property
    def colormap(self):
        """tuple of str, vispy.color.Colormap: colormap for luminance images.
        """
        return self._colormap_name, self._node.cmap

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
            name = _increment_unnamed_colormap(
                name, list(self._colormaps.keys())
            )
            self._colormaps[name] = colormap
        else:
            warn(f'invalid value for colormap: {colormap}')
            name = self._colormap_name
        self._colormap_name = name
        self._node.cmap = self._colormaps[name]
        self._update_thumbnail()
        self.events.colormap()

    @property
    def colormaps(self):
        """tuple of str: names of available colormaps.
        """
        return tuple(self._colormaps.keys())

    # wrap visual properties:
    @property
    def clim(self):
        """list of float: Limits to use for the colormap.
        Can be 'auto' to auto-set bounds to the min and max of the data.
        """
        return list(self._node.clim)

    @clim.setter
    def clim(self, clim):
        self._clim_msg = f'{float(clim[0]): 0.3}, {float(clim[1]): 0.3}'
        self.status = self._clim_msg
        self._node.clim = clim
        if clim[0] < self._clim_range[0]:
            self._clim_range[0] = copy(clim[0])
        if clim[1] > self._clim_range[1]:
            self._clim_range[1] = copy(clim[1])
        self._update_thumbnail()
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
        """{
            'bessel', 'bicubic', 'bilinear', 'blackman', 'catrom', 'gaussian',
            'hamming', 'hanning', 'hermite', 'kaiser', 'lanczos', 'mitchell',
            'nearest', 'spline16', 'spline36'
            }: Equipped interpolation method's name.

        """
        return str(self._interpolation)

    @interpolation.setter
    def interpolation(self, interpolation):
        if isinstance(interpolation, str):
            interpolation = Interpolation(interpolation)
        self._interpolation = interpolation
        self._node.interpolation = interpolation.value
        self.events.interpolation()

    def _clim_range_default(self):
        min = self.data.min()
        max = self.data.max()
        if min == max:
            min = 0
            max = 1
        return [float(min), float(max)]

    def _update_thumbnail(self):
        """Update thumbnail with current image data and colormap.
        """
        image = self._data_thumbnail
        zoom_factor = np.divide(
            self._thumbnail_shape[:2], image.shape[:2]
        ).min()
        if self.multichannel:
            downsampled = ndi.zoom(
                image, (zoom_factor, zoom_factor, 1), prefilter=False, order=0
            )
            if image.shape[2] == 4:  # image is RGBA
                colormapped[..., 3] = downsampled[..., 3] * self.opacity
            else:  # image is RGB
                alpha = np.full(
                    downsampled.shape[:2] + (1,),
                    int(255 * self.opacity),
                    dtype=np.uint8,
                )
                colormapped = np.concatenate([downsampled, alpha], axis=2)
            colormapped = colormapped.astype(np.uint8)
        else:
            downsampled = ndi.zoom(
                image, zoom_factor, prefilter=False, order=0
            )
            low, high = self.clim
            downsampled = np.clip(downsampled, low, high)
            color_range = high - low
            if color_range != 0:
                downsampled = (downsampled - low) / color_range
            colormapped = self.colormap[1].map(downsampled)
            colormapped = colormapped.reshape(downsampled.shape + (4,))
            colormapped[..., 3] *= self.opacity
        self.thumbnail = colormapped

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
        if self.multichannel:
            shape = self._data_view.shape[:-1]
        else:
            shape = self._data_view.shape
        coord[-2:] = np.clip(coord[-2:], 0, np.asarray(shape) - 1)

        value = self._data_view[tuple(coord[-2:])]

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
        image = np.clip(self._data_view, self.clim[0], self.clim[1])
        image = image - self.clim[0]
        color_range = self.clim[1] - self.clim[0]
        if color_range != 0:
            image = image / color_range
        mapped_image = (self.colormap[1].map(image) * 255).astype('uint8')
        mapped_image = mapped_image.reshape(list(self._data_view.shape) + [4])
        image_str = imwrite('<bytes>', mapped_image, format='png')
        image_str = "data:image/png;base64," + str(b64encode(image_str))[2:-1]
        props = {'xlink:href': image_str}
        width = str(self.shape[-1])
        height = str(self.shape[-2])
        opacity = str(self.opacity)
        xml = Element(
            'image', width=width, height=height, opacity=opacity, **props
        )
        return [xml]

    def on_mouse_move(self, event):
        """Called whenever mouse moves over canvas.
        """
        if event.pos is None:
            return
        self.position = tuple(event.pos)
        coord, value = self.get_value()
        self.status = self.get_message(coord, value)
