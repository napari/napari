import warnings
from enum import Enum

import numpy as np
from pydantic import root_validator, validator

from ..events import EventedModel
from ..events.custom_types import Array
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
    interpolation: ColormapInterpolationMode = ColormapInterpolationMode.LINEAR
    controls: Array[float, (-1,)] = None

    def __init__(self, colors, **data):
        super().__init__(colors=colors, **data)

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
        if not np.array_equal(v, sorted(v)):
            raise ValueError("Coords needs to be sorted in ascending order")
        return v

    @root_validator(skip_on_failure=True)
    def check_bound_coords(cls, values):
        colors = values['colors']
        controls = values['controls']
        if controls[0] != 0:
            warnings.warn(
                f"colormap need to have first coord equal to 0, not {controls[0]}",
                RuntimeWarning,
            )
            controls = np.concatenate(([0], controls))
            colors = np.concatenate(([colors[0]], colors))
        if controls[-1] != 1:
            warnings.warn(
                f"colormap need to have last coord equal to 1, not {controls[-1]}",
                RuntimeWarning,
            )
            controls = np.concatenate((controls, [1]))
            colors = np.concatenate((colors, [colors[-1]]))
        if controls.size != colors.shape[0] + int(
            values['interpolation'] == ColormapInterpolationMode.ZERO
        ):
            raise ValueError(
                "Number of colors does not match with length of controls"
            )
        if not np.array_equal(controls, sorted(controls)):
            # To find examples like [-2, 0.5 ,7]
            raise ValueError("Coords needs to be in range [0, 1]")
        values['colors'] = colors
        values['controls'] = controls
        return values

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
            raise ValueError('Unrecognized Colormap Interpolation Mode')

        return cols

    @property
    def colorbar(self):
        return make_colorbar(self)
