import bisect
import math
from collections import defaultdict
from functools import cached_property, lru_cache
from typing import Any, DefaultDict, Dict, List, Optional, Tuple, cast

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


class LabelColormapBase(Colormap):
    _cache_mapping: Dict[Tuple[np.dtype, np.dtype], np.ndarray] = PrivateAttr(
        default={}
    )
    _cache_other: Dict[str, Any] = PrivateAttr(default={})

    class Config(Colormap.Config):
        # this config is to avoid deepcopy of cached_property
        # see https://github.com/pydantic/pydantic/issues/2763
        # it is required until drop pydantic 1 or pythin 3.11 and older
        # need to validate after drop pydantic 1
        keep_untouched = (cached_property,)

    def _cmap_without_selection(self) -> "LabelColormapBase":
        if self.use_selection:
            cmap = self.__class__(**self.dict())
            cmap.use_selection = False
            return cmap
        return self

    def _get_mapping_from_cache(
        self, data_dtype: np.dtype
    ) -> Optional[np.ndarray]:
        target_dtype = _texture_dtype(self._num_unique_colors, data_dtype)
        key = (data_dtype, target_dtype)
        if key not in self._cache_mapping and data_dtype.itemsize <= 2:
            data = np.arange(
                np.iinfo(target_dtype).max + 1, dtype=target_dtype
            ).astype(data_dtype)
            self._cache_mapping[key] = self._map_without_cache(data)
        return self._cache_mapping.get(key)

    def _clear_cache(self):
        """Mechanism to clean cached properties"""
        self._cache_mapping = {}
        self._cache_other = {}

    @property
    def _num_unique_colors(self) -> int:
        """Number of unique colors, not counting transparent black."""
        return len(self.colors) - 1

    def _map_without_cache(self, values: np.ndarray) -> np.ndarray:
        """Function that maps values to colors without selection or cache"""
        raise NotImplementedError

    def _selection_as_minimum_dtype(self, dtype: np.dtype) -> int:
        """Treat selection as given dtype and calculate value with min dtype.

        Parameters
        ----------
        dtype : np.dtype
            The dtype to convert the selection to.

        Returns
        -------
        int
            The selection converted.
        """
        raise NotImplementedError


class LabelColormap(LabelColormapBase):
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

    def _selection_as_minimum_dtype(self, dtype: np.dtype) -> int:
        return int(
            _cast_labels_data_to_texture_dtype_auto(
                np.array([self.selection]).astype(dtype), self
            )[0]
        )

    def _background_as_minimum_dtype(self, dtype: np.dtype) -> int:
        """Treat background as given dtype and calculate value with min dtype.

        Parameters
        ----------
        dtype : np.dtype
            The dtype to convert the background to.

        Returns
        -------
        int
            The background converted.
        """
        return int(
            _cast_labels_data_to_texture_dtype_auto(
                np.array([self.background_value]).astype(dtype), self
            )[0]
        )

    def _map_without_cache(self, values) -> np.ndarray:
        texture_dtype_values = _zero_preserving_modulo_numpy(
            values,
            len(self.colors) - 1,
            values.dtype,
            self.background_value,
        )
        mapped = self.colors[texture_dtype_values]
        mapped[texture_dtype_values == 0] = 0
        return mapped

    def map(self, values) -> np.ndarray:
        """Map values to colors.

        Parameters
        ----------
        values : np.ndarray or float
            Values to be mapped.

        Returns
        -------
        np.ndarray of same shape as values, but with last dimension of size 4
            Mapped colors.
        """
        values = np.atleast_1d(values)

        if values.dtype.kind == 'f':
            values = values.astype(np.int64)
        mapper = self._get_mapping_from_cache(values.dtype)
        if mapper is not None:
            mapped = mapper[values]
        else:
            mapped = self._map_without_cache(values)
        if self.use_selection:
            mapped[(values != self.selection)] = 0

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


class DirectLabelColormap(LabelColormapBase):
    """Colormap using a direct mapping from labels to color using a dict.

    Attributes
    ----------
    color_dict: dict from int to (3,) or (4,) array
        The dictionary mapping labels to colors.
    use_selection : bool
        Whether to color using the selected label.
    selection : int
        The selected label.
    """

    color_dict: DefaultDict[Optional[int], np.ndarray] = Field(
        default_factory=lambda: defaultdict(lambda: np.zeros(4))
    )
    use_selection: bool = False
    selection: int = 0

    def __init__(self, *args, **kwargs) -> None:
        if "colors" not in kwargs and not args:
            kwargs["colors"] = np.zeros(3)
        super().__init__(*args, **kwargs)

    def _selection_as_minimum_dtype(self, dtype: np.dtype) -> int:
        return int(
            _cast_labels_data_to_texture_dtype_direct(
                np.array([self.selection]).astype(dtype), self
            )[0]
        )

    def _get_hash_cache(self, data_dtype: np.dtype) -> np.ndarray:
        key = f"_{data_dtype}_hash_cache"
        if key not in self._cache_other:
            self._cache_other[key] = _generate_hash_map_for_direct_colormap(
                self._cmap_without_selection(), data_dtype
            )
        return self._cache_other[key]

    def map(self, values) -> np.ndarray:
        """
        Map values to colors.
        Parameters
        ----------
        values : np.ndarray or int
            Values to be mapped.

        Returns
        -------
        np.ndarray of same shape as values, but with last dimension of size 4
            Mapped colors.
        """
        values = np.atleast_1d(values)
        if values.dtype.kind in {'f', 'U'}:
            raise TypeError("DirectLabelColormap can only be used with int")
        mapper = self._get_mapping_from_cache(values.dtype)
        if mapper is not None:
            mapped = mapper[values]
        else:
            casted = _cast_labels_data_to_texture_dtype_direct_impl(
                values, self
            )
            mapped = self._map_casted(casted, apply_selection=True)

        if self.use_selection:
            mapped[(values != self.selection)] = 0
        return mapped

    def _map_without_cache(self, values: np.ndarray) -> np.ndarray:
        cmap = self._cmap_without_selection()
        casted = _cast_labels_data_to_texture_dtype_direct_impl(values, cmap)
        return self._map_casted(casted, apply_selection=False)

    def _map_casted(self, values, apply_selection) -> np.ndarray:
        """
        Map values to colors.
        Parameters
        ----------
        values : np.ndarray
            Values to be mapped. It need to be already casted using
            cast_labels_to_minimum_type_auto
        Returns
        -------
        np.ndarray of shape (N, M, 4)
            Mapped colors.
        Notes
        -----
        it is implemented for thumbnail labels,
        where we already have casted values
        """
        mapped = np.zeros(values.shape + (4,), dtype=np.float32)
        colors = self._values_mapping_to_minimum_values_set(apply_selection)[1]
        for idx in np.ndindex(values.shape):
            value = values[idx]
            mapped[idx] = colors[value]
        return mapped

    @cached_property
    def _num_unique_colors(self) -> int:
        """Count the number of unique colors in the colormap."""
        return len({tuple(x) for x in self.color_dict.values()})

    def _clear_cache(self):
        super()._clear_cache()
        if "_num_unique_colors" in self.__dict__:
            del self.__dict__["_num_unique_colors"]
        if "_label_mapping_and_color_dict" in self.__dict__:
            del self.__dict__["_label_mapping_and_color_dict"]

    def _values_mapping_to_minimum_values_set(
        self, apply_selection=True
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
        if self.use_selection and apply_selection:
            return {self.selection: 1, None: 0}, {
                0: np.array((0, 0, 0, 0)),
                1: self.color_dict.get(
                    self.selection,
                    self.default_color,
                ),
            }

        return self._label_mapping_and_color_dict

    @cached_property
    def _label_mapping_and_color_dict(
        self,
    ) -> Tuple[Dict[Optional[int], int], Dict[int, np.ndarray]]:
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


def _convert_small_ints_to_unsigned(data: np.ndarray) -> np.ndarray:
    """Convert (u)int8 to uint8 and (u)int16 to uint16.

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


def _cast_labels_data_to_texture_dtype_auto(
    data: np.ndarray,
    colormap: LabelColormap,
) -> np.ndarray:
    """Convert labels data to the data type used in the texture.

    In https://github.com/napari/napari/issues/6397, we noticed that using
    float32 textures was much slower than uint8 or uint16 textures. Here we
    convert the labels data to uint8 or uint16, based on the following rules:

    - uint8 and uint16 labels data are unchanged. (No copy of the arrays.)
    - int8 and int16 data are converted with a *view* to uint8 and uint16.
      (This again does not involve a copy so is fast, and lossless.)
    - higher precision integer data (u)int{32,64} are hashed to uint8, uint16,
      or float32, depending on the number of colors in the input colormap. (See
      `minimum_dtype_for_labels`.) Since the hashing can result in collisions,
      this conversion *has* to happen in the CPU to correctly map the
      background and selection values.

    Parameters
    ----------
    data : np.ndarray
        Labels data to be converted.
    colormap : LabelColormap
        Colormap used to display the labels data.

    Returns
    -------
    np.ndarray
        Converted labels data.
    """
    if data.itemsize <= 2:
        return _convert_small_ints_to_unsigned(data)

    num_colors = len(colormap.colors) - 1

    dtype = minimum_dtype_for_labels(num_colors + 1)

    if colormap.use_selection:
        selection_in_texture = _zero_preserving_modulo(
            np.array([colormap.selection]), num_colors, dtype
        )
        converted = np.where(
            data == colormap.selection, selection_in_texture, dtype.type(0)
        )
    else:
        converted = _zero_preserving_modulo(
            data, num_colors, dtype, colormap.background_value
        )

    return converted


def _zero_preserving_modulo_numpy(
    values: np.ndarray, n: int, dtype: np.dtype, to_zero: int = 0
) -> np.ndarray:
    """``(values - 1) % n + 1``, but with one specific value mapped to 0.

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
    res = ((values - 1) % n + 1).astype(dtype)
    res[values == to_zero] = 0
    return res


def _zero_preserving_modulo_jit(
    values: np.ndarray, n: int, dtype: np.dtype, to_zero: int = 0
) -> np.ndarray:
    """``(values - 1) % n + 1``, but with one specific value mapped to 0.

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

    for i in prange(values.size):
        if values.flat[i] == to_zero:
            result.flat[i] = 0
        else:
            result.flat[i] = (values.flat[i] - 1) % n + 1

    return result


def _cast_labels_data_to_texture_dtype_direct(
    data: np.ndarray, direct_colormap: DirectLabelColormap
) -> np.ndarray:
    data = _convert_small_ints_to_unsigned(data)

    if data.itemsize <= 2:
        return data

    return _cast_labels_data_to_texture_dtype_direct_impl(
        data, direct_colormap
    )


def _cast_labels_data_to_texture_dtype_direct_numpy(
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
    max_value = max(x for x in direct_colormap.color_dict if x is not None)
    if max_value > 2**16:
        raise RuntimeError(  # pragma: no cover
            "Cannot use numpy implementation for large values of labels "
            "direct colormap. Please install numba."
        )
    dtype = minimum_dtype_for_labels(direct_colormap._num_unique_colors + 2)
    label_mapping = direct_colormap._values_mapping_to_minimum_values_set()[0]

    mapper = np.full((max_value + 2), DEFAULT_VALUE, dtype=dtype)
    for key, val in label_mapping.items():
        if key is None:
            continue
        mapper[key] = val

    if data.dtype.itemsize > 2:
        data = np.clip(data, 0, max_value + 1)
    return mapper[data]


def _generate_hash_map_for_direct_colormap(
    direct_colormap: DirectLabelColormap,
    data_dtype: np.dtype,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate hash map for direct colormap.
    """
    target_dtype = minimum_dtype_for_labels(
        direct_colormap._num_unique_colors + 2
    )
    label_mapping = direct_colormap._values_mapping_to_minimum_values_set()[0]
    prime_num_array = _primes(upto=2**16)
    pos = bisect.bisect_left(prime_num_array, len(label_mapping) * 2)
    if pos < len(prime_num_array):
        hash_size = prime_num_array[pos]
    else:
        hash_size = 2 ** (math.ceil(math.log2(len(label_mapping))) + 1)

    hash_table_key = np.zeros(hash_size, dtype=data_dtype)
    hash_table_val = np.zeros(hash_size, dtype=target_dtype)

    data_min = np.iinfo(data_dtype).min
    data_max = np.iinfo(data_dtype).max

    for key, val in label_mapping.items():
        if key is None:
            continue
        if key > data_max or key < data_min:
            continue
        new_key = key % hash_size
        while hash_table_key[new_key] != 0:
            new_key = (new_key + 1) % hash_size

        hash_table_key[new_key] = key
        hash_table_val[new_key] = val

    return hash_table_key, hash_table_val


def _generate_hash_map_and_cast_labels_data_to_texture_dtype_direct(
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
        The cast data array.
    """
    if not direct_colormap.use_selection:
        hash_table_key, hash_table_val = direct_colormap._get_hash_cache(
            data.dtype
        )
    else:
        (
            hash_table_key,
            hash_table_val,
        ) = _generate_hash_map_for_direct_colormap(direct_colormap, data.dtype)

    return _cast_direct_labels_to_minimum_type_jit(
        data, hash_table_key, hash_table_val
    )


def _cast_direct_labels_to_minimum_type_jit(
    data: np.ndarray,
    hash_table_key: np.ndarray,
    hash_table_val: np.ndarray,
) -> np.ndarray:
    result_array = np.zeros_like(data, dtype=hash_table_val.dtype)

    # iterate over data and calculate modulo num_colors assigning to result_array

    hash_size = hash_table_key.size

    for i in prange(data.size):
        key = data.flat[i]
        new_key = int(key % hash_size)
        while hash_table_key[new_key] != key:
            if hash_table_key[new_key] == 0:
                result_array.flat[i] = DEFAULT_VALUE
                break
            # This will stop because half of the hash table is empty
            new_key = (new_key + 1) % hash_size
        else:
            result_array.flat[i] = hash_table_val[new_key]

    return result_array


def _texture_dtype(num_colors: int, dtype: np.dtype) -> np.dtype:
    """Compute VisPy texture dtype given number of colors and raw data dtype.

    - for data of type int8 and uint8 we can use uint8 directly, with no copy.
    - for int16 and uint16 we can use uint16 with no copy.
    - for any other dtype, we fall back on `minimum_dtype_for_labels`, which
      will require on-CPU mapping between the raw data and the texture dtype.
    """
    if dtype.itemsize == 1:
        return np.dtype(np.uint8)
    if dtype.itemsize == 2:
        return np.dtype(np.uint16)
    return minimum_dtype_for_labels(num_colors)


def minimum_dtype_for_labels(num_colors: int) -> np.dtype:
    """Return the minimum texture dtype that can hold given number of colors.

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


try:
    import numba
except ModuleNotFoundError:
    _zero_preserving_modulo = _zero_preserving_modulo_numpy
    _cast_labels_data_to_texture_dtype_direct_impl = (
        _cast_labels_data_to_texture_dtype_direct_numpy
    )
    prange = range
else:
    _zero_preserving_modulo = numba.njit(parallel=True)(
        _zero_preserving_modulo_jit
    )
    _cast_labels_data_to_texture_dtype_direct_impl = (
        _generate_hash_map_and_cast_labels_data_to_texture_dtype_direct
    )
    _cast_direct_labels_to_minimum_type_jit = numba.njit(parallel=True)(
        _cast_direct_labels_to_minimum_type_jit
    )
    prange = numba.prange  # type: ignore [misc]

    del numba


@lru_cache(maxsize=128)
def _primes(upto):
    """Generate primes up to a given number.
    Parameters
    ----------
    upto : int
        The upper limit of the primes to generate.
    Returns
    -------
    primes : np.ndarray
        The primes up to the upper limit.
    """
    primes = np.arange(3, upto + 1, 2)
    isprime = np.ones((upto - 1) // 2, dtype=bool)
    max_factor = int(np.sqrt(upto))
    for factor in primes[: max_factor // 2]:
        if isprime[(factor - 2) // 2]:
            isprime[(factor * 3 - 2) // 2 : None : factor] = 0
    return np.concatenate(([2], primes[isprime]))
