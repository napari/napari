import warnings

import numpy as np
from vispy.color import Colormap

from ..utils.colormaps import make_colorbar
from ..utils.event import Event
from ..utils.status_messages import format_float
from ..utils.validators import validate_n_seq
from .layer_utils import increment_unnamed_colormap


validate_2_tuple = validate_n_seq(2)


class IntensityVisualizationMixin:
    """A mixin that adds gamma, colormap, and contrast limits logic to Layers.

    When used, this should come before the Layer in the inheritance, e.g.:

        class Image(ImageSurfaceMixin, Layer):
            def __init__(self):
                ...
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.events.add(contrast_limits=Event, gamma=Event, colormap=Event)
        self._gamma = 1
        self._colormap_name = ''
        self._contrast_limits_msg = ''
        self._contrast_limits = [None, None]
        self._contrast_limits_range = [None, None]

    def reset_contrast_limits(self):
        """Scale contrast limits to data range"""
        data_range = self._calc_data_range()
        self.contrast_limits = data_range

    def reset_contrast_limits_range(self):
        """Scale contrast limits range to data type.

        Currently, this only does something if the data type is an unsigned
        integer... otherwise it's unclear what the full range should be.
        """
        if np.issubdtype(self.dtype, np.unsignedinteger):
            info = np.iinfo(self.dtype)
            self.contrast_limits_range = (info.min, info.max)

    @property
    def colormap(self):
        """2-tuple of str, vispy.color.Colormap: colormap for luminance images.
        """
        return self._colormap_name, self._cmap

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
        elif isinstance(colormap, Colormap):
            name = increment_unnamed_colormap(
                name, list(self._colormaps.keys())
            )
            self._colormaps[name] = colormap
        else:
            warnings.warn(f'invalid value for colormap: {colormap}')
            name = self._colormap_name
        self._colormap_name = name
        self._cmap = self._colormaps[name]
        self._colorbar = make_colorbar(self._cmap)
        self._update_thumbnail()
        self.events.colormap()

    @property
    def colormaps(self):
        """tuple of str: names of available colormaps."""
        return tuple(self._colormaps.keys())

    @property
    def contrast_limits(self):
        """list of float: Limits to use for the colormap."""
        return list(self._contrast_limits)

    @contrast_limits.setter
    def contrast_limits(self, contrast_limits):
        self.events.contrast_limits(
            name="contrast_limits", value=contrast_limits
        )

    def _set_contrast_limits(self, contrast_limits):
        validate_2_tuple(contrast_limits)
        self._contrast_limits_msg = (
            format_float(contrast_limits[0])
            + ', '
            + format_float(contrast_limits[1])
        )
        self.status = self._contrast_limits_msg
        self._contrast_limits = contrast_limits
        # make sure range slider is big enough to fit range
        newrange = list(self.contrast_limits_range)
        newrange[0] = min(newrange[0], contrast_limits[0])
        newrange[1] = max(newrange[1], contrast_limits[1])
        self.contrast_limits_range = newrange
        self._update_thumbnail()

    @property
    def contrast_limits_range(self):
        """The current valid range of the contrast limits."""
        return list(self._contrast_limits_range)

    @contrast_limits_range.setter
    def contrast_limits_range(self, value):
        """Set the valid range of the contrast limits"""
        validate_2_tuple(value)
        if list(value) == self.contrast_limits_range:
            return

        # if either value is "None", it just preserves the current range
        current_range = self.contrast_limits_range
        value = list(value)  # make sure it is mutable
        for i in range(2):
            value[i] = current_range[i] if value[i] is None else value[i]
        self._contrast_limits_range = value

        # make sure that the current values fit within the new range
        # this also serves the purpose of emitting events.contrast_limits()
        # and updating the views/controllers
        if hasattr(self, '_contrast_limits') and any(self._contrast_limits):
            cur_min, cur_max = self.contrast_limits
            new_min = min(max(value[0], cur_min), value[1])
            new_max = max(min(value[1], cur_max), value[0])
            self.contrast_limits = (new_min, new_max)
            self.events.contrast_limits()

    @property
    def gamma(self):
        return self._gamma

    @gamma.setter
    def gamma(self, value):
        self.status = format_float(value)
        self._gamma = value
        self._update_thumbnail()
        self.events.gamma()
