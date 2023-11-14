import bisect
import math
from collections import defaultdict
from typing import DefaultDict, Dict, List, Optional, Tuple, cast

import numba
import numpy as np

from napari._pydantic_compat import Field, PrivateAttr, validator
from napari.utils.color import ColorArray
from napari.utils.colormaps.colorbars import make_colorbar
from napari.utils.compat import StrEnum
from napari.utils.events import EventedModel
from napari.utils.events.custom_types import Array
from napari.utils.translations import trans

DEFAULT_VALUE = 0


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
    _display_name : str
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
                    'Control points must start with 0.0 and end with 1.0. '
                    'Got {start_control_point} and {end_control_point}',
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
    background_value: int = 0

    def map(self, values):
        values = np.atleast_1d(values)

        mapped = self.colors[
            cast_labels_to_minimum_type_auto(
                values, len(self.colors) - 1, self.background_value
            ).astype(np.int64)
        ]

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
    color_dict: dict from int to (3,) or (4,) array
        The dictionary mapping labels to colors.
    use_selection: bool
        Whether to color using the selected label.
    selection: float
        The selected label.
    """

    color_dict: DefaultDict[Optional[int], np.ndarray] = Field(
        default_factory=lambda: defaultdict(lambda: np.zeros(4))
    )
    use_selection: bool = False
    selection: int = 0

    def map(self, values):
        # Convert to float32 to match the current GL shader implementation
        values = np.atleast_1d(values)
        casted = cast_direct_labels_to_minimum_type(values, self)
        return self.map_casted(casted)

    def map_casted(self, values):
        mapped = np.zeros(values.shape + (4,), dtype=np.float32)
        colors = self.values_mapping_to_minimum_values_set()[1]
        for idx in np.ndindex(values.shape):
            value = values[idx]
            mapped[idx] = colors[value]
        return mapped

    def unique_colors_num(self) -> int:
        """Count the number of unique colors in the colormap."""
        return len({tuple(x) for x in self.color_dict.values()})

    def values_mapping_to_minimum_values_set(
        self,
    ) -> Tuple[Dict[Optional[int], int], Dict[int, np.ndarray]]:
        """Create mapping from original values to minimum values set.
        To use minimum possible dtype for labels.

        Returns
        -------
        Dict[Optional[int], int]
            Mapping from original values to minimum values set.
        Dict[int, np.ndarray]
            Mapping from new values to colors.

        """
        if self.use_selection:
            return {self.selection: 1, None: 0}, {
                0: np.array((0, 0, 0, 0)),
                1: self.color_dict.get(
                    self.selection,
                    self.default_color,
                ),
            }

        color_to_labels: Dict[Tuple[int, ...], List[Optional[int]]] = {}
        labels_to_new_labels: Dict[Optional[int], int] = {None: 0}
        new_color_dict: Dict[int, np.ndarray] = {
            DEFAULT_VALUE: self.default_color,
        }

        for label, color in self.color_dict.items():
            if label is None:
                continue
            color_tup = tuple(color)
            if color_tup not in color_to_labels:
                color_to_labels[color_tup] = [label]
                labels_to_new_labels[label] = len(new_color_dict)
                new_color_dict[labels_to_new_labels[label]] = color
            else:
                color_to_labels[color_tup].append(label)
                labels_to_new_labels[label] = labels_to_new_labels[
                    color_to_labels[color_tup][0]
                ]

        return labels_to_new_labels, new_color_dict

    @property
    def default_color(self) -> np.ndarray:
        return self.color_dict.get(None, np.array((0, 0, 0, 0)))
        # we provided here default color for backward compatibility
        # if someone is using DirectLabelColormap directly, not through Label layer


def cast_labels_to_minimum_type_auto(
    data: np.ndarray, num_colors: int, background_value: int
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
    dtype = minimum_dtype_for_labels(num_colors + 1)

    return _modulo_plus_one(data, num_colors, dtype, background_value)


@numba.njit(parallel=True)
def _modulo_plus_one(
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
            result.flat[i] = values.flat[i] % n + 1

    return result


def cast_direct_labels_to_minimum_type(
    data: np.ndarray, direct_colormap: DirectLabelColormap
) -> np.ndarray:
    """
    Cast direct labels to the minimum type.

    Parameters
    ----------
    data : np.ndarray
        The input data array.
    direct_colormap : DirectLabelColormap
        The direct colormap.

    Returns
    -------
    np.ndarray
        The casted data array.
    """

    dtype = minimum_dtype_for_labels(direct_colormap.unique_colors_num() + 2)

    label_mapping = direct_colormap.values_mapping_to_minimum_values_set()[0]
    pos = bisect.bisect_left(PRIME_NUM_TABLE, len(label_mapping) * 2)
    if pos < len(PRIME_NUM_TABLE):
        hash_size = PRIME_NUM_TABLE[pos]
    else:
        hash_size = 2 ** (math.ceil(math.log2(len(label_mapping))) + 1)

    hash_table_key = np.zeros(hash_size, dtype=np.uint64)
    hash_table_val = np.zeros(hash_size, dtype=dtype)

    for key, val in label_mapping.items():
        if key is None:
            continue
        new_key = key % hash_size
        while hash_table_key[new_key] != 0:
            new_key = (new_key + 1) % hash_size

        hash_table_key[new_key] = key
        hash_table_val[new_key] = val

    return _cast_direct_labels_to_minimum_type_auto(
        data, hash_table_key, hash_table_val, dtype
    )


@numba.njit(parallel=True)
def _cast_direct_labels_to_minimum_type_auto(
    data: np.ndarray,
    hash_table_key: np.ndarray,
    hash_table_val: np.ndarray,
    dtype: np.dtype,
) -> np.ndarray:
    result_array = np.zeros_like(data, dtype=dtype)

    # iterate over data and calculate modulo num_colors assigning to result_array

    hash_size = hash_table_key.size

    for i in numba.prange(data.size):
        key = data.flat[i]
        new_key = int(key % hash_size)
        while hash_table_key[new_key] != key:
            if hash_table_key[new_key] == 0:
                result_array.flat[i] = DEFAULT_VALUE
                break
            # This will stop because half of the hash table is empty
            new_key = (new_key + 1) % hash_size
        result_array.flat[i] = hash_table_val[new_key]

    return result_array


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


PRIME_NUM_TABLE = [
    37,
    61,
    127,
    251,
    509,
    1021,
    2039,
    4093,
    8191,
    16381,
    32749,
    65521,
]
