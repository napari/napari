from enum import Enum
from typing import Optional

import numpy as np
from pydantic import PrivateAttr, validator

from napari.utils.color import ColorArray

from ..events import EventedModel
from ..events.custom_types import Array
from ..translations import trans
from .colorbars import make_colorbar


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
    colors: ColorArray
    name: str = 'custom'
    _display_name: Optional[str] = PrivateAttr(None)
    interpolation: ColormapInterpolationMode = ColormapInterpolationMode.LINEAR
    controls: Array[float, (-1,)] = None

    def __init__(self, colors, display_name: Optional[str] = None, **data):
        if display_name is None:
            display_name = data.get('name', 'custom')

        super().__init__(colors=colors, **data)
        self._display_name = display_name

    # controls validator must be called even if None for correct initialization
    @validator('controls', pre=True, always=True)
    def _check_controls(cls, v, values):
        # If no control points provided generate defaults
        if v is None or len(v) == 0:
            n_controls = len(values.get('colors', [])) + int(
                values['interpolation'] == ColormapInterpolationMode.ZERO
            )
            return np.linspace(0, 1, n_controls)

        # Check control end points are correct
        if v[0] != 0 or (len(v) > 1 and v[-1] != 1):
            raise ValueError(
                trans._(
                    'Control points must start with 0.0 and end with 1.0. Got {start_control_point} and {end_control_point}',
                    deferred=True,
                    start_control_point=v[0],
                    end_control_point=v[-1],
                )
            )

        # Check control points are sorted correctly
        if not np.array_equal(v, sorted(v)):
            raise ValueError(
                trans._(
                    'Control points need to be sorted in ascending order',
                    deferred=True,
                )
            )

        # Check number of control points is correct
        n_controls_target = len(values['colors']) + int(
            values['interpolation'] == ColormapInterpolationMode.ZERO
        )
        n_controls = len(v)
        if n_controls != n_controls_target:
            raise ValueError(
                trans._(
                    'Wrong number of control points provided. Expected {n_controls_target}, got {n_controls}',
                    deferred=True,
                    n_controls_target=n_controls_target,
                    n_controls=n_controls,
                )
            )

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
