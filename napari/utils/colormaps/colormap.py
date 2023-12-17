from collections import defaultdict
from functools import cached_property
from typing import (
    TYPE_CHECKING,
    Any,
    DefaultDict,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    cast,
    overload,
)

import numpy as np

from napari._pydantic_compat import Field, PrivateAttr, validator
from napari.utils.color import ColorArray
from napari.utils.colormaps.colorbars import make_colorbar
from napari.utils.compat import StrEnum
from napari.utils.events import EventedModel
from napari.utils.events.custom_types import Array
from napari.utils.translations import trans

if TYPE_CHECKING:
    from numba import typed

MAPPING_OF_UNKNOWN_VALUE = 0
# For direct mode we map all unknown values to single value
# for simplicity of implementation we select 0


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
    use_selection: bool = False
    selection: int = 0
    background_value: int = 0
    interpolation: ColormapInterpolationMode = ColormapInterpolationMode.ZERO
    _cache_mapping: Dict[Tuple[np.dtype, np.dtype], np.ndarray] = PrivateAttr(
        default={}
    )
    _cache_other: Dict[str, Any] = PrivateAttr(default={})

    class Config(Colormap.Config):
        # this config is to avoid deepcopy of cached_property
        # see https://github.com/pydantic/pydantic/issues/2763
        # it is required until we drop Pydantic 1 or Python 3.11 and older
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
        """For given dtype, return precomputed array mapping values to colors.

        Returns None if the dtype itemsize is greater than 2.
        """
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

    def _selection_as_minimum_dtype(self, dtype: np.dtype) -> int:
        return int(
            _cast_labels_data_to_texture_dtype_auto(
                dtype.type(self.selection), self
            )
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
                dtype.type(self.background_value), self
            )
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
                dtype.type(self.selection), self
            )
        )

    def map(self, values) -> np.ndarray:
        """Map values to colors.

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
            values_cast = _labels_raw_to_texture_direct(values, self)
            mapped = self._map_precast(values_cast, apply_selection=True)

        if self.use_selection:
            mapped[(values != self.selection)] = 0
        return mapped

    def _map_without_cache(self, values: np.ndarray) -> np.ndarray:
        cmap = self._cmap_without_selection()
        cast = _labels_raw_to_texture_direct(values, cmap)
        return self._map_precast(cast, apply_selection=False)

    def _map_precast(self, values, apply_selection) -> np.ndarray:
        """Map values to colors.

        Parameters
        ----------
        values : np.ndarray
            Values to be mapped. It need to be already cast using
            cast_labels_to_minimum_type_auto

        Returns
        -------
        np.ndarray of shape (N, M, 4)
            Mapped colors.

        Notes
        -----
        it is implemented for thumbnail labels,
        where we already have cast values
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
        if "_array_map" in self.__dict__:
            del self.__dict__["_array_map"]

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
        labels_to_new_labels: Dict[Optional[int], int] = {
            None: MAPPING_OF_UNKNOWN_VALUE
        }
        new_color_dict: Dict[int, np.ndarray] = {
            MAPPING_OF_UNKNOWN_VALUE: self.default_color,
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

    def _get_typed_dict_mapping(self, data_dtype: np.dtype) -> 'typed.Dict':
        """Create mapping from label values to texture values of smaller dtype.

        In https://github.com/napari/napari/issues/6397, we noticed that using
        float32 textures was much slower than uint8 or uint16 textures. When
        labels data is (u)int(8,16), we simply use the labels data directly.
        But when it is higher-precision, we need to compress the labels into
        the smallest dtype that can still achieve the goal of the
        visualisation. This corresponds to the smallest dtype that can map to
        the number of unique colors in the colormap. Even if we have a
        million labels, if they map to one of two colors, we can map them to
        a uint8 array with values 1 and 2; then, the texture can map those
        two values to each of the two possible colors.

        Returns
        -------
        Dict[Optional[int], int]
            Mapping from original values to minimal texture value set.
        """

        # we cache the result to avoid recomputing it on each slice;
        # check first if it's already in the cache.
        key = f"_{data_dtype}_typed_dict"
        if key in self._cache_other:
            return self._cache_other[key]

        from numba import typed, types

        # num_unique_colors + 2 because we need to map None and background
        target_type = minimum_dtype_for_labels(self._num_unique_colors + 2)

        dkt = typed.Dict.empty(
            key_type=getattr(types, data_dtype.name),
            value_type=getattr(types, target_type.name),
        )
        for k, v in self._label_mapping_and_color_dict[0].items():
            if k is None:
                continue
            dkt[data_dtype.type(k)] = target_type.type(v)

        self._cache_other[key] = dkt

        return dkt

    @cached_property
    def _array_map(self):
        """Create an array to map labels to texture values of smaller dtype."""
        max_value = max(x for x in self.color_dict if x is not None)
        if max_value > 2**16:
            raise RuntimeError(  # pragma: no cover
                "Cannot use numpy implementation for large values of labels "
                "direct colormap. Please install numba."
            )
        dtype = minimum_dtype_for_labels(self._num_unique_colors + 2)
        label_mapping = self._values_mapping_to_minimum_values_set()[0]

        # We need 2 + the max value: one because we will be indexing with the
        # max value, and an extra one so that higher values get clipped to
        # that index and map to the default value, rather than to the max
        # value in the map.
        mapper = np.full(
            (max_value + 2), MAPPING_OF_UNKNOWN_VALUE, dtype=dtype
        )
        for key, val in label_mapping.items():
            if key is None:
                continue
            mapper[key] = val
        return mapper

    @property
    def default_color(self) -> np.ndarray:
        return self.color_dict.get(None, np.array((0, 0, 0, 0)))
        # we provided here default color for backward compatibility
        # if someone is using DirectLabelColormap directly, not through Label layer


def _convert_small_ints_to_unsigned(
    data: Union[np.ndarray, np.integer],
) -> Union[np.ndarray, np.integer]:
    """Convert (u)int8 to uint8 and (u)int16 to uint16.

    Otherwise, return the original array.

    Parameters
    ----------
    data : np.ndarray | np.integer
        Data to be converted.

    Returns
    -------
    np.ndarray | np.integer
        Converted data.
    """
    if data.dtype.itemsize == 1:
        # for fast rendering of int8
        return data.view(np.uint8)
    if data.dtype.itemsize == 2:
        # for fast rendering of int16
        return data.view(np.uint16)
    return data


@overload
def _cast_labels_data_to_texture_dtype_auto(
    data: np.ndarray,
    colormap: LabelColormap,
) -> np.ndarray:
    ...


@overload
def _cast_labels_data_to_texture_dtype_auto(
    data: np.integer,
    colormap: LabelColormap,
) -> np.integer:
    ...


def _cast_labels_data_to_texture_dtype_auto(
    data: Union[np.ndarray, np.integer],
    colormap: LabelColormap,
) -> Union[np.ndarray, np.integer]:
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
    np.ndarray | np.integer
        Converted labels data.
    """
    original_shape = np.shape(data)
    if data.itemsize <= 2:
        return _convert_small_ints_to_unsigned(data)

    data_arr = np.atleast_1d(data)
    num_colors = len(colormap.colors) - 1

    dtype = minimum_dtype_for_labels(num_colors + 1)

    if colormap.use_selection:
        selection_in_texture = _zero_preserving_modulo(
            np.array([colormap.selection]), num_colors, dtype
        )
        converted = np.where(
            data_arr == colormap.selection, selection_in_texture, dtype.type(0)
        )
    else:
        converted = _zero_preserving_modulo(
            data_arr, num_colors, dtype, colormap.background_value
        )

    return np.reshape(converted, original_shape)


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


def _zero_preserving_modulo_loop(
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
    # need to preallocate numpy array for asv memory benchmarks
    return _zero_preserving_modulo_inner_loop(values, n, to_zero, out=result)


def _zero_preserving_modulo_inner_loop(
    values: np.ndarray, n: int, to_zero: int, out: np.ndarray
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
    to_zero : int
        A specific value to map to 0. (Usually, 0 itself.)
    out : np.ndarray
        Preallocated output array

    Returns
    -------
    np.ndarray
        The result: 0 for the ``to_zero`` value, ``values % n + 1``
        everywhere else.
    """
    for i in prange(values.size):
        if values.flat[i] == to_zero:
            out.flat[i] = 0
        else:
            out.flat[i] = (values.flat[i] - 1) % n + 1

    return out


@overload
def _cast_labels_data_to_texture_dtype_direct(
    data: np.ndarray, direct_colormap: DirectLabelColormap
) -> np.ndarray:
    ...


@overload
def _cast_labels_data_to_texture_dtype_direct(
    data: np.integer, direct_colormap: DirectLabelColormap
) -> np.integer:
    ...


def _cast_labels_data_to_texture_dtype_direct(
    data: Union[np.ndarray, np.integer], direct_colormap: DirectLabelColormap
) -> Union[np.ndarray, np.integer]:
    """Convert labels data to the data type used in the texture.

    In https://github.com/napari/napari/issues/6397, we noticed that using
    float32 textures was much slower than uint8 or uint16 textures. Here we
    convert the labels data to uint8 or uint16, based on the following rules:

    - uint8 and uint16 labels data are unchanged. (No copy of the arrays.)
    - int8 and int16 data are converted with a *view* to uint8 and uint16.
      (This again does not involve a copy so is fast, and lossless.)
    - higher precision integer data (u)int{32,64} are mapped to an intermediate
      space of sequential values based on the colors they map to. As an
      example, if the values are [1, 2**25, and 2**50], and the direct
      colormap maps them to ['red', 'green', 'red'], then the intermediate map
      is {1: 1, 2**25: 2, 2**50: 1}. The labels can then be converted to a
      uint8 texture and a smaller direct colormap with only two values.

    This function calls `_labels_raw_to_texture_direct`, but makes sure that
    signed ints are first viewed as their unsigned counterparts.

    Parameters
    ----------
    data : np.ndarray | np.integer
        Labels data to be converted.
    direct_colormap : LabelColormap
        Colormap used to display the labels data.

    Returns
    -------
    np.ndarray | np.integer
        Converted labels data.
    """
    data = _convert_small_ints_to_unsigned(data)

    if data.itemsize <= 2:
        return data

    original_shape = np.shape(data)
    array_data = np.atleast_1d(data)
    return np.reshape(
        _labels_raw_to_texture_direct(array_data, direct_colormap),
        original_shape,
    )


def _labels_raw_to_texture_direct_numpy(
    data: np.ndarray, direct_colormap: DirectLabelColormap
) -> np.ndarray:
    """Convert labels data to the data type used in the texture.

    This implementation uses numpy vectorized operations.

    See `_cast_labels_data_to_texture_dtype_direct` for more details.
    """
    mapper = direct_colormap._array_map

    if data.dtype.itemsize > 2:
        data = np.clip(data, 0, mapper.shape[0] - 1)
    return mapper[data]


def _labels_raw_to_texture_direct_loop(
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
    if direct_colormap.use_selection:
        return (data == direct_colormap.selection).astype(np.uint8)

    dkt = direct_colormap._get_typed_dict_mapping(data.dtype)
    target_dtype = minimum_dtype_for_labels(
        direct_colormap._num_unique_colors + 2
    )
    result_array = np.full_like(
        data, MAPPING_OF_UNKNOWN_VALUE, dtype=target_dtype
    )
    return _labels_raw_to_texture_direct_inner_loop(data, dkt, result_array)


def _labels_raw_to_texture_direct_inner_loop(
    data: np.ndarray, dkt: 'typed.Dict', out: np.ndarray
) -> np.ndarray:
    """
    Relabel data using typed dict with mapping unknown labels to default value
    """
    # The numba typed dict does not provide official Api for
    # determine key and value types
    for i in prange(data.size):
        val = data.flat[i]
        if val in dkt:
            out.flat[i] = dkt[data.flat[i]]

    return out


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
    _labels_raw_to_texture_direct = _labels_raw_to_texture_direct_numpy
    prange = range
else:
    _zero_preserving_modulo_inner_loop = numba.njit(parallel=True)(
        _zero_preserving_modulo_inner_loop
    )
    _zero_preserving_modulo = _zero_preserving_modulo_loop
    _labels_raw_to_texture_direct = _labels_raw_to_texture_direct_loop
    _labels_raw_to_texture_direct_inner_loop = numba.njit(parallel=True)(
        _labels_raw_to_texture_direct_inner_loop
    )
    prange = numba.prange  # type: ignore [misc]

    del numba
