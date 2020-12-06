from enum import auto
from itertools import cycle
from typing import Any, Dict, Union

import numpy as np

from ...utils.misc import StringEnum
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


class Colormap:
    """Colormap that relates intensity values to colors.

    Parameters
    ----------
    colors : array, shape (N, 4)
        Data used in the colormap.
    controls : array, shape (N,) or (N+1,)
        Control points of the colormap.
    interpolation : str
        Colormap interpolation mode, either 'linear' or
        'zero'. If 'linear', ncontrols = ncolors (one
        color per control point). If 'zero', ncontrols
        = ncolors+1 (one color per bin).
    name : str
        Name of the colormap.
    """

    def __init__(
        self, colors, *, controls=None, interpolation='linear', name='custom'
    ):

        self.name = name
        self.colors = transform_color(colors)
        self._interpolation = ColormapInterpolationMode(interpolation)
        if controls is None:
            n_controls = len(self.colors) + int(
                self._interpolation == ColormapInterpolationMode.ZERO
            )
            self.controls = np.linspace(0, 1, n_controls)
        else:
            self.controls = np.asarray(controls)

    def __iter__(self):
        yield from (self.colors, self.controls, str(self.interpolation))

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

    @property
    def interpolation(self):
        return str(self._interpolation)


class CategoricalColormap:
    """Colormap that relates categorical values to colors.

    Parameters
    ----------
    colormap : array, shape (N, 4)
        Data used in the colormap.

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
            self._colormap = colormap
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
        color_properties = np.asarray(color_properties)

        # add properties if they are not in the colormap
        color_cycle_keys = [*self.colormap]
        props_in_map = np.in1d(color_properties, color_cycle_keys)
        if not np.all(props_in_map):
            indices_to_add = np.unique(
                color_properties[np.logical_not(props_in_map)],
                return_index=True,
            )[1]
            props_to_add = [
                color_properties[index] for index in sorted(indices_to_add)
            ]
            for prop in props_to_add:
                new_color = next(self._fallback_color)
                self.colormap[prop] = np.squeeze(transform_color(new_color))
        # map the colors
        colors = np.array([self.colormap[x] for x in color_properties])
        return colors

    # def _map_value(self, value: Any) -> np.ndarray:
    #     if value in self.colormap:
    #         color = self.colormap[value]
    #     else:
    #         if isinstance(self.fallback_color, cycle):
    #             color = next(self.fallback_color)
    #         else:
    #             color = self.fallback_color
    #         self.colormap[value] = color
    #     return color
