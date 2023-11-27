from collections import defaultdict
from functools import cached_property
from typing import Optional, cast

import numpy as np

from napari._pydantic_compat import Field, PrivateAttr, validator
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
    selection : int
    """

    seed: float = 0.5
    use_selection: bool = False
    selection: int = 0
    interpolation: ColormapInterpolationMode = ColormapInterpolationMode.ZERO
    background_value: int = 0

    class Config:
        keep_untouched = (cached_property,)

    @cached_property
    def _uint8_colors(self) -> np.ndarray:
        data = np.arange(256, dtype=np.uint8)
        return self.map(data, apply_selection=False)

    @cached_property
    def _uint16_colors(self) -> np.ndarray:
        data = np.arange(65536, dtype=np.uint16)
        return self.map(data, apply_selection=False)

    def map(self, values, apply_selection=True) -> np.ndarray:
        """Map values to colors.

        Parameters
        ----------
        values : np.ndarray or float
            Values to be mapped.
        apply_selection : bool
            Whether to apply selection if self.use_selection is True.

        Returns
        -------
        np.ndarray of same shape as values, but with last dimension of size 4
            Mapped colors.
        """
        values = np.atleast_1d(values)

        if values.dtype.kind == 'f':
            values = values.astype(np.int64)

        if values.dtype == np.uint8 and "_uint8_colors" in self.__dict__:
            mapped = self._uint8_colors[values]
        elif values.dtype == np.uint16 and "_uint16_colors" in self.__dict__:
            mapped = self._uint16_colors[values]
        else:
            background = int(
                np.array([self.background_value]).astype(values.dtype)[0]
            )
            casted_values = _zero_preserving_modulo_naive(
                values, len(self.colors) - 1, values.dtype, background
            )
            mapped = self.colors[casted_values]
            mapped[casted_values == 0] = 0
        if self.use_selection and apply_selection:
            selection2 = np.array([self.selection]).astype(values.dtype)[0]
            mapped[(values != self.selection) & (values != selection2)] = 0

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
    color_dict : defaultdict
        The dictionary mapping labels to colors.
    use_selection : bool
        Whether to color using the selected label.
    selection : float
        The selected label.
    """

    color_dict: defaultdict = Field(
        default_factory=lambda: defaultdict(lambda: np.zeros(4))
    )
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
            return 0, 0, 0, 0
        return self.color_dict.get(None, (0, 0, 0, 0))
        # we provided here default color for backward compatibility
        # if someone is using DirectLabelColormap directly, not through Label layer


def _convert_small_ints_to_unsigned(data: np.ndarray) -> np.ndarray:
    """
    Convert int8 to uint8 and int16 to uint16.
    Otherwise, return the original array.

    Parameters
    ----------
    data : np.ndarray
        Data to be converted.

    Returns
    -------
    np.ndarray
        Converted data.
    """
    if data.dtype.itemsize == 1:
        # for fast rendering of int8
        return data.view(np.uint8)
    if data.dtype.itemsize == 2:
        # for fast rendering of int16
        return data.view(np.uint16)
    return data


def _cast_labels_to_minimum_dtype_auto(
    data: np.ndarray,
    colormap: LabelColormap,
) -> np.ndarray:
    """Perform modulo operation based on number of colors

    Parameters
    ----------
    data : np.ndarray
        Labels data to be casted.
    num_colors : int
        Number of unique colors in the data.
    background_value : int
        The value in ``values`` to be treated as the background.

    Returns
    -------
    np.ndarray
        Casted labels data.
    """
    data = _convert_small_ints_to_unsigned(data)

    if data.itemsize <= 2:
        return data

    num_colors = len(colormap.colors) - 1

    dtype = minimum_dtype_for_labels(num_colors + 1)

    if colormap.use_selection:
        return (data == colormap.selection).astype(dtype) * (
            _zero_preserving_modulo_naive(
                np.array([colormap.selection]), num_colors, dtype
            )
        )

    return _zero_preserving_modulo(
        data, num_colors, dtype, colormap.background_value
    )


def _zero_preserving_modulo_naive(
    values: np.ndarray, n: int, dtype: np.dtype, to_zero: int = 0
) -> np.ndarray:
    res = ((values - 1) % n + 1).astype(dtype)
    res[values == to_zero] = 0
    return res


try:
    import numba
except ImportError:
    _zero_preserving_modulo = _zero_preserving_modulo_naive
else:

    @numba.njit(parallel=True)
    def _zero_preserving_modulo(
        values: np.ndarray, n: int, dtype: np.dtype, to_zero: int = 0
    ) -> np.ndarray:
        """Like ``values % n + 1``, but with one specific value mapped to 0.

        This ensures (1) an output value in [0, n] (inclusive), and (2) that
        no nonzero values in the input are zero in the output, other than the
        ``to_zero`` value.

        Parameters
        ----------
        values : np.ndarray
            The dividend of the modulo operator.
        n : int
            The divisor.
        dtype : np.dtype
            The desired dtype for the output array.
        to_zero : int, optional
            A specific value to map to 0. (By default, 0 itself.)

        Returns
        -------
        np.ndarray
            The result: 0 for the ``to_zero`` value, ``values % n + 1``
            everywhere else.
        """
        result = np.empty_like(values, dtype=dtype)

        for i in numba.prange(values.size):
            if values.flat[i] == to_zero:
                result.flat[i] = 0
            else:
                result.flat[i] = (values.flat[i] - 1) % n + 1

        return result


def minimum_dtype_for_labels(num_colors: int) -> np.dtype:
    """Return the minimum dtype that can hold the number of colors.

    Parameters
    ----------
    num_colors : int
        Number of unique colors in the data.

    Returns
    -------
    np.dtype
        Minimum dtype that can hold the number of colors.
    """
    if num_colors <= np.iinfo(np.uint8).max:
        return np.dtype(np.uint8)
    if num_colors <= np.iinfo(np.uint16).max:
        return np.dtype(np.uint16)
    return np.dtype(np.float32)
