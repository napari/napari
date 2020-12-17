from enum import auto
from itertools import cycle
from typing import Any, Dict, Union

import numpy as np

from ..events.dataclass import Property, evented_dataclass
from ..misc import StringEnum
from .color_transformations import transform_color_cycle
from .colorbars import make_colorbar
from .standardize_color import transform_color


class ColormapInterpolationMode(StringEnum):
    """INTERPOLATION: Interpolation mode for colormaps.

    Selects an interpolation mode for the colormap.
            * linear: colors are defined by linear interpolation between
              colors of neighboring controls points.
            * zero: colors are defined by the value of the color in the
              bin between by neighboring controls points.
    """

    LINEAR = auto()
    ZERO = auto()


@evented_dataclass
class Colormap:
    """Colormap that relates intensity values to colors.

    Attributes
    ----------
    colors : array, shape (N, 4)
        Data used in the colormap.
    name : str
        Name of the colormap.
    controls : array, shape (N,) or (N+1,)
        Control points of the colormap.
    interpolation : str
        Colormap interpolation mode, either 'linear' or
        'zero'. If 'linear', ncontrols = ncolors (one
        color per control point). If 'zero', ncontrols
        = ncolors+1 (one color per bin).
    """

    colors: Property[np.ndarray, None, transform_color]
    name: str = 'custom'
    controls: Property[np.ndarray, None, np.asarray] = np.zeros((0, 4))
    interpolation: Property[
        ColormapInterpolationMode, str, ColormapInterpolationMode
    ] = ColormapInterpolationMode.LINEAR

    def __post_init__(self):
        if len(self.controls) == 0:
            n_controls = len(self.colors) + int(
                self._interpolation == ColormapInterpolationMode.ZERO
            )
            self.controls = np.linspace(0, 1, n_controls)

    def __iter__(self):
        yield from (self.colors, self.controls, self.interpolation)

    def map(self, values):
        values = np.atleast_1d(values)
        if self._interpolation == ColormapInterpolationMode.LINEAR:
            # One color per control point
            cols = [
                np.interp(values, self.controls, self.colors[:, i])
                for i in range(4)
            ]
            cols = np.stack(cols, axis=1)
        elif self._interpolation == ColormapInterpolationMode.ZERO:
            # One color per bin
            indices = np.clip(
                np.searchsorted(self.controls, values) - 1, 0, len(self.colors)
            )
            cols = self.colors[indices.astype(np.int32)]
        else:
            raise ValueError('Unrecognized Colormap Interpolation Mode')

        return cols

    @property
    def colorbar(self):
        return make_colorbar(self)


class CategoricalColormap:
    """Colormap that relates categorical values to colors.

    Parameters
    ----------
    colormap : dict
        The mapping between categorical property values and color.
    fallback_color : cycle
        The color to be used in the case that a value is mapped that is not
        in colormap. This can be given as any ColorType and it will be converted
        to a cycle. The default value is a cycle of all black.

    """

    def __init__(
        self, colormap: Union[dict, list, cycle], fallback_color='black'
    ):
        self.fallback_color = fallback_color
        self.colormap = colormap

    @property
    def colormap(self) -> Dict[Any, np.ndarray]:
        return self._colormap

    @colormap.setter
    def colormap(self, colormap):
        if isinstance(colormap, list) or isinstance(colormap, np.ndarray):
            self.fallback_color = colormap

            # reset the color mapping
            self._colormap = {}
        elif isinstance(colormap, dict):
            transformed_colormap = {
                k: transform_color(v)[0] for k, v in colormap.items()
            }
            self._colormap = transformed_colormap
        else:
            raise TypeError('colormap should be an array or dict')

    @property
    def fallback_color(self) -> np.ndarray:
        return self._fallback_color_values

    @fallback_color.setter
    def fallback_color(self, fallback_color):
        (transformed_color_cycle, transformed_colors,) = transform_color_cycle(
            color_cycle=fallback_color,
            elem_name='color_cycle',
            default="white",
        )
        self._fallback_color = transformed_color_cycle
        self._fallback_color_values = transformed_colors

    def map(self, color_properties: Union[list, np.ndarray]) -> np.ndarray:
        """Map an array of values to an array of colors

        Parameters
        ----------
        color_properties : Union[list, np.ndarray]
            The property values to be converted to colors.

        Returns
        -------
        colors : np.ndarray
            An Nx4 color array where N is the number of property values provided.
        """
        if isinstance(color_properties, (list, np.ndarray)):
            color_properties = np.asarray(color_properties)
        else:
            color_properties = np.asarray([color_properties])

        # add properties if they are not in the colormap
        color_cycle_keys = [*self.colormap]
        props_in_map = np.in1d(color_properties, color_cycle_keys)
        if not np.all(props_in_map):
            new_prop_values = color_properties[np.logical_not(props_in_map)]
            indices_to_add = np.unique(new_prop_values, return_index=True)[1]
            props_to_add = [
                new_prop_values[index] for index in sorted(indices_to_add)
            ]
            for prop in props_to_add:
                new_color = next(self._fallback_color)
                self.colormap[prop] = np.squeeze(transform_color(new_color))
        # map the colors
        colors = np.array([self.colormap[x] for x in color_properties])
        return colors
