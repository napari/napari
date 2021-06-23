from enum import Enum
from typing import Optional

import numpy as np
from pydantic import PrivateAttr, validator

from ..events import EventedModel
from ..events.custom_types import Array
from ..translations import trans
from .colorbars import make_colorbar
from .standardize_color import transform_color


class ColormapInterpolationMode(str, Enum):
    """INTERPOLATION: Interpolation mode for colormaps.

    Selects an interpolation mode for the colormap.
            * linear: colors are defined by linear interpolation between
              colors of neighboring controls points.
            * zero: colors are defined by the value of the color in the
              bin between by neighboring controls points.
    """

    LINEAR = 'linear'
    ZERO = 'zero'


class Colormap(EventedModel):
    """Colormap that relates intensity values to colors.

    Attributes
    ----------
    colors : array, shape (N, 4)
        Data used in the colormap.
    name : str
        Name of the colormap.
    display_name : str
        Display name of the colormap.
    controls : array, shape (N,) or (N+1,)
        Control points of the colormap.
    interpolation : str
        Colormap interpolation mode, either 'linear' or
        'zero'. If 'linear', ncontrols = ncolors (one
        color per control point). If 'zero', ncontrols
        = ncolors+1 (one color per bin).
    """

    # fields
    colors: Array[float, (-1, 4)]
    name: str = 'custom'
    _display_name: Optional[str] = PrivateAttr(None)
    interpolation: ColormapInterpolationMode = ColormapInterpolationMode.LINEAR
    controls: Array[float, (-1,)] = None

    def __init__(self, colors, display_name: Optional[str] = None, **data):
        if display_name is None:
            display_name = data.get('name', 'custom')

        super().__init__(colors=colors, **data)
        self._display_name = display_name

    # validators
    @validator('colors', pre=True)
    def _ensure_color_array(cls, v):
        return transform_color(v)

    # controls validator must be called even if None for correct initialization
    @validator('controls', pre=True, always=True)
    def _check_controls(cls, v, values):
        if v is None or len(v) == 0:
            n_controls = len(values['colors']) + int(
                values['interpolation'] == ColormapInterpolationMode.ZERO
            )
            return np.linspace(0, 1, n_controls)
        return v

    def __iter__(self):
        yield from (self.colors, self.controls, self.interpolation)

    def map(self, values):
        values = np.atleast_1d(values)
        if self.interpolation == ColormapInterpolationMode.LINEAR:
            # One color per control point
            cols = [
                np.interp(values, self.controls, self.colors[:, i])
                for i in range(4)
            ]
            cols = np.stack(cols, axis=1)
        elif self.interpolation == ColormapInterpolationMode.ZERO:
            # One color per bin
            indices = np.clip(
                np.searchsorted(self.controls, values) - 1, 0, len(self.colors)
            )
            cols = self.colors[indices.astype(np.int32)]
        else:
            raise ValueError(
                trans._(
                    'Unrecognized Colormap Interpolation Mode',
                    deferred=True,
                )
            )

        return cols

    @property
    def colorbar(self):
        return make_colorbar(self)
