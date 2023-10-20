from collections import defaultdict
from typing import Optional, cast

import numpy as np
from pydantic import Field, PrivateAttr, validator

from napari.utils.color import ColorArray
from napari.utils.colormaps.colorbars import make_colorbar
from napari.utils.compat import StrEnum
from napari.utils.events import EventedModel
from napari.utils.events.custom_types import Array
from napari.utils.translations import trans


class ColormapInterpolationMode(StrEnum):
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
    controls: Array = Field(default_factory=lambda: cast(Array, []))

    def __init__(
        self, colors, display_name: Optional[str] = None, **data
    ) -> None:
        if display_name is None:
            display_name = data.get('name', 'custom')

        super().__init__(colors=colors, **data)
        self._display_name = display_name

    # controls validator must be called even if None for correct initialization
    @validator('controls', pre=True, always=True, allow_reuse=True)
    def _check_controls(cls, v, values):
        # If no control points provided generate defaults
        if v is None or len(v) == 0:
            n_controls = len(values['colors']) + int(
                values['interpolation'] == ColormapInterpolationMode.ZERO
            )
            return np.linspace(0, 1, n_controls, dtype=np.float32)

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
            cols = np.stack(cols, axis=-1)
        elif self.interpolation == ColormapInterpolationMode.ZERO:
            # One color per bin
            # Colors beyond max clipped to final bin
            indices = np.clip(
                np.searchsorted(self.controls, values, side="right") - 1,
                0,
                len(self.colors) - 1,
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


class LabelColormap(Colormap):
    """Colormap that shuffles values before mapping to colors.

    Attributes
    ----------
    seed : float
    use_selection : bool
    selection : float
    """

    seed: float = 0.5
    use_selection: bool = False
    selection: float = 0.0
    interpolation: ColormapInterpolationMode = ColormapInterpolationMode.ZERO
    background_value: float = 0.0

    def map(self, values):
        values = np.atleast_1d(values)

        mapped = self.colors[np.mod(values, len(self.colors)).astype(np.int64)]

        mapped[values == self.background_value] = 0

        # If using selected, disable all others
        if self.use_selection:
            mapped[~np.isclose(values, self.selection)] = 0

        return mapped

    def shuffle(self, seed: int):
        """Shuffle the colormap colors.

        Parameters
        ----------
        seed : int
            Seed for the random number generator.
        """
        np.random.default_rng(seed).shuffle(self.colors[1:])
        self.events.colors(value=self.colors)


class DirectLabelColormap(Colormap):
    """Colormap using a direct mapping from labels to color using a dict.

    Attributes
    ----------
    color_dict: defaultdict
        The dictionary mapping labels to colors.
    use_selection: bool
        Whether to color using the selected label.
    selection: float
        The selected label.
    """

    color_dict: defaultdict = defaultdict(lambda: np.zeros(4))
    use_selection: bool = False
    selection: float = 0.0

    def map(self, values):
        # Convert to float32 to match the current GL shader implementation
        values = np.atleast_1d(values).astype(np.float32)
        mapped = np.zeros(values.shape + (4,), dtype=np.float32)
        for idx in np.ndindex(values.shape):
            value = values[idx]
            if value in self.color_dict:
                color = self.color_dict[value]
                if len(color) == 3:
                    color = np.append(color, 1)
                mapped[idx] = color
            else:
                mapped[idx] = self.default_color
        # If using selected, disable all others
        if self.use_selection:
            mapped[~np.isclose(values, self.selection)] = 0
        return mapped

    @property
    def default_color(self):
        if self.use_selection:
            return (0, 0, 0, 0)
        return self.color_dict.get(None, (0, 0, 0, 0))
        # we provided here default color for backward compatibility
        # if someone is using DirectLabelColormap directly, not through Label layer
