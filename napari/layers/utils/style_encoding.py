from abc import ABC, abstractmethod
from string import Formatter
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import numpy as np
from pydantic import Field, ValidationError, parse_obj_as, validator

if TYPE_CHECKING:
    from pydantic.typing import ReprArgs

from ...utils import Colormap
from ...utils.colormaps import ValidColormapArg, ensure_colormap
from ...utils.colormaps.categorical_colormap import CategoricalColormap
from ...utils.colormaps.standardize_color import transform_color
from ...utils.events import EventedModel


class StyleEncoding(EventedModel, ABC):
    """Defines a way to encode style values, like colors and strings.

    This also updates and stores values generated using that encoding.

    Attributes
    ----------
    array : np.ndarray
        Stores the generated style values. The first dimension should have
        length N, where N is the number of expected style values. The other
        dimensions should describe the dimensionality of the style value.
    """

    array: np.ndarray = []

    @abstractmethod
    def _apply(
        self, properties: Dict[str, np.ndarray], indices: Sequence[int]
    ) -> np.ndarray:
        pass

    def validate_properties(self, properties: Dict[str, np.ndarray]):
        """Validates that the given properties are compatible with this encoding.

        Parameters
        ----------
        properties : Dict[str, np.ndarray]
            The properties of a layer.

        Raises
        ------
        ValueError
            If the given properties are not compatible with this encoding.
        """
        pass

    def update_all(self, properties: Dict[str, np.ndarray], n_rows: int):
        """Updates all style values based on the given properties.

        Parameters
        ----------
        properties : Dict[str, np.ndarray]
            The properties of a layer.
        n_rows : int
            The total number of rows in the properties table, which should
            correspond to the number of elements in a layer, and will be the
            number of style values generated.
        """
        indices = range(0, n_rows)
        self.array = self._apply(properties, indices)

    def update_tail(self, properties: Dict[str, np.ndarray], n_rows: int):
        """Generates style values for newly added elements in properties and appends them to this.

        Parameters
        ----------
        properties : Dict[str, np.ndarray]
            The properties of a layer.
        n_rows : int
            The total number of rows in the properties table, which should
            correspond to the number of elements in a layer, and will be the
            number of style values generated.
        """
        n_values = self.array.shape[0]
        indices = range(n_values, n_rows)
        array = self._apply(properties, indices)
        self.append(array)

    def append(self, array: np.ndarray):
        """Appends raw style values to this.

        This is useful for supporting the paste operation in layers.

        Parameters
        ----------
        array : np.ndarray
            The values to append. The dimensionality of these should match that of the existing style values.
        """
        self.array = _append_maybe_empty(self.array, array)

    def delete(self, indices: Iterable[int]):
        """Deletes style values from this by index.

        Parameters
        ----------
        indices : Iterable[int]
            The indices of the style values to remove.
        """
        self.array = np.delete(self.array, list(indices), axis=0)

    @validator('array', pre=True, always=True)
    def _check_array(cls, array):
        return np.array(array)


class DerivedStyleEncoding(StyleEncoding, ABC):
    """Encodes style values from properties.

    The style values can always be regenerated from properties, so the
    array attribute should read, but not written for persistent storage,
    as updating this type of encoding will clobber the written values.
    For the same reason, the array attribute is not serialized.
    """

    def __repr_args__(self) -> 'ReprArgs':
        return [
            (name, value)
            for name, value in super().__repr_args__()
            if name != 'array'
        ]

    def json(self, **kwargs):
        exclude = _add_to_exclude(kwargs.pop('exclude', None), 'array')
        return super().json(exclude=exclude, **kwargs)

    def dict(self, **kwargs):
        exclude = _add_to_exclude(kwargs.pop('exclude', None), 'array')
        return super().dict(exclude=exclude, **kwargs)


class DirectStyleEncoding(StyleEncoding):
    """Encodes style values directly.

    The style values are encoded directly in the array attribute, so that
    attribute can be written to make persistent updates.

    Attributes
    ----------
    default : np.ndarray
        The default style value that is used when requesting a value that
        is out of bounds in the array attribute. In general this is a numpy
        array because color is a 1D RGBA numpy array, but mostly this will
        be a 0D numpy array (i.e. a scalar).
    """

    default: np.ndarray

    def _apply(
        self, properties: Dict[str, np.ndarray], indices: Sequence[int]
    ) -> np.ndarray:
        n_values = self.array.shape[0]
        in_bound_indices = [index for index in indices if index < n_values]
        n_default = len(indices) - len(in_bound_indices)
        return _append_maybe_empty(
            self.array[in_bound_indices],
            np.array([self.default] * n_default),
        )


class DirectColorEncoding(DirectStyleEncoding):
    """Encodes color values directly in an array."""

    @validator('array', pre=True, always=True)
    def _check_array(cls, array):
        return np.empty((0, 4)) if len(array) == 0 else transform_color(array)

    @validator('default', pre=True, always=True)
    def _check_default(cls, default):
        return transform_color(default)[0]


class ConstantColorEncoding(DerivedStyleEncoding):
    """Encodes color values from a constant.

    Attributes
    ----------
    constant : np.ndarray
        The color that is always returned regardless of property values.
        Can be provided as any of the types in :class:`ColorType`, but will
        be coerced to an RGBA numpy array of shape (4,).
    """

    constant: np.ndarray

    def _apply(
        self, properties: Dict[str, np.ndarray], indices: Sequence[int]
    ) -> np.ndarray:
        return np.tile(self.constant, (len(indices), 1))

    @validator('constant', pre=True, always=True)
    def _check_constant(cls, constant) -> np.ndarray:
        return transform_color(constant)[0]


class IdentityColorEncoding(DerivedStyleEncoding):
    """Encodes color values directly from a property column.

    Attributes
    ----------
    property_name : str
        The name of the property that contains the desired color values.
    """

    property_name: str = Field(..., allow_mutation=False)

    def _apply(
        self, properties: Dict[str, np.ndarray], indices: Sequence[int]
    ) -> np.ndarray:
        return transform_color(properties[self.property_name][indices])

    def validate_properties(self, properties: Dict[str, np.ndarray]):
        _check_property_name(properties, self.property_name)


class DiscreteColorEncoding(DerivedStyleEncoding):
    """Encodes color values from a discrete property column whose values are mapped to colors.

    Attributes
    ----------
    property_name : str
        The name of the property that contains the discrete values from which to map.
    categorical_colormap : CategoricalColormap
        Maps the selected property values to colors.
    """

    property_name: str = Field(..., allow_mutation=False)
    categorical_colormap: CategoricalColormap

    def _apply(
        self, properties: Dict[str, np.ndarray], indices: Sequence[int]
    ) -> np.ndarray:
        values = properties[self.property_name][indices]
        return self.categorical_colormap.map(values)

    def validate_properties(self, properties: Dict[str, np.ndarray]):
        _check_property_name(properties, self.property_name)


class ContinuousColorEncoding(DerivedStyleEncoding):
    """Encodes color values from a continuous property column whose values are mapped to colors.

    Attributes
    ----------
    property_name : str
        The name of the property that contains the continuous values from which to map.
    continuous_colormap : Colormap
        Maps the selected property values to colors.
    contrast_limits : Optional[Tuple[float, float]]
        The (min, max) property values that should respectively map to the first and last
        colors in the colormap. If None, then this will attempt to calculate these values
        from the property values the first time this generate color values. If that attempt
        fails, these are effectively (0, 1).
    """

    property_name: str = Field(..., allow_mutation=False)
    continuous_colormap: Colormap
    contrast_limits: Optional[Tuple[float, float]] = None

    def _apply(
        self, properties: Dict[str, np.ndarray], indices: Sequence[int]
    ) -> np.ndarray:
        all_values = properties[self.property_name]
        if self.contrast_limits is None:
            self.contrast_limits = _calculate_contrast_limits(all_values)
        values = all_values[indices]
        if self.contrast_limits is not None:
            values = np.interp(values, self.contrast_limits, (0, 1))
        return self.continuous_colormap.map(values)

    @validator('continuous_colormap', pre=True, always=True)
    def _check_continuous_colormap(
        cls, colormap: ValidColormapArg
    ) -> Colormap:
        return ensure_colormap(colormap)

    @validator('contrast_limits', pre=True, always=True)
    def _check_contrast_limits(
        cls, contrast_limits
    ) -> Optional[Tuple[float, float]]:
        if (contrast_limits is not None) and (
            contrast_limits[0] >= contrast_limits[1]
        ):
            raise ValueError(
                'contrast_limits must be a strictly increasing pair of values'
            )
        return contrast_limits

    def validate_properties(self, properties: Dict[str, np.ndarray]):
        _check_property_name(properties, self.property_name)


class ConstantStringEncoding(DerivedStyleEncoding):
    """Encodes string values directly in an array.

    Attributes
    ----------
    constant : str
        The string that is always returned regardless of property values.
    """

    constant: str

    def _apply(
        self, properties: Dict[str, np.ndarray], indices: Sequence[int]
    ) -> np.ndarray:
        return np.repeat(self.constant, len(indices))


class FormatStringEncoding(DerivedStyleEncoding):
    """Encodes string values by formatting property values.

    Attributes
    ----------
    format_string : str
        A format string with the syntax supported by :func:`str.format`,
        where all format fields should be property names.
    """

    format_string: str = Field(..., allow_mutation=False)

    def _apply(
        self, properties: Dict[str, np.ndarray], indices: Sequence[int]
    ) -> np.ndarray:
        return np.array(
            [
                self.format_string.format(
                    **_get_property_row(properties, index)
                )
                for index in indices
            ]
        )

    def validate_properties(self, properties: Dict[str, np.ndarray]):
        is_format_string(properties, self.format_string)


class DirectStringEncoding(DirectStyleEncoding):
    """Encodes string values directly in an array."""

    @validator('array', pre=True, always=True)
    def _check_array(cls, array) -> np.ndarray:
        return np.array(array, dtype=str)

    @validator('default', pre=True, always=True)
    def _check_default(cls, default) -> np.ndarray:
        return np.array(default, dtype=str)


def parse_kwargs_as_encoding(encodings: Tuple[type, ...], **kwargs):
    """Parses the given kwargs as one of the given encodings.

    Parameters
    ----------
    encodings : Tuple[type, ...]
        The supported encoding types, each of which must be a subclass of
        :class:`StyleEncoding`. The first encoding that can be constructed
        from the given kwargs will be returned.

    Raises
    ------
    ValueError
        If the provided kwargs cannot be used to construct any of the given encodings.
    """
    try:
        return parse_obj_as(Union[encodings], kwargs)
    except ValidationError as error:
        raise ValueError(
            'Failed to parse a supported encoding from kwargs:\n'
            f'{kwargs}\n\n'
            'The kwargs must specify the fields of exactly one of the following encodings:\n'
            f'{encodings}\n\n'
            'Original error:\n'
            f'{error}'
        )


def is_format_string(
    properties: Dict[str, np.ndarray], format_string: str
) -> bool:
    """Returns true if the given string can be used in :class:`StringFormatEncoding`.

    Parameters
    ----------
    properties : Dict[str, np.ndarray]
        The properties of a layer.
    format_string : str
        The format string.

    Returns
    -------
    True if format_string contains at least one field, False otherwise.

    Raises
    ------
    ValueError
        If the format_string is not valid (e.g. mismatching braces), or one of the
        format fields is not a property name.
    """
    fields = tuple(
        field
        for _, field, _, _ in Formatter().parse(format_string)
        if field is not None
    )
    for field in fields:
        if field not in properties:
            raise ValueError(
                f'Found format string field {field} without a corresponding property'
            )
    return len(fields) > 0


# Define supported encodings as tuples instead of Union, so that they can be used with
# isinstance without relying on get_args, which was only added in python 3.8.
# The order of the encoding matters because we rely on Pydantic for parsing dict kwargs
# inputs for each encoding - it will use the first encoding that can be defined with those kwargs.

"""The color encodings supported by napari in order of precedence."""
COLOR_ENCODINGS = (
    ContinuousColorEncoding,
    DiscreteColorEncoding,
    ConstantColorEncoding,
    IdentityColorEncoding,
    DirectColorEncoding,
)

"""The string encodings supported by napari in order of precedence."""
STRING_ENCODINGS = (
    FormatStringEncoding,
    ConstantStringEncoding,
    DirectStringEncoding,
)


def _append_maybe_empty(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if right.size == 0:
        return left
    if left.size == 0:
        return right
    return np.append(left, right, axis=0)


def _calculate_contrast_limits(
    values: np.ndarray,
) -> Optional[Tuple[float, float]]:
    contrast_limits = None
    if values.size > 0:
        min_value = np.min(values)
        max_value = np.max(values)
        # Use < instead of != to handle nans.
        if min_value < max_value:
            contrast_limits = (min_value, max_value)
    return contrast_limits


def _get_property_row(
    properties: Dict[str, np.ndarray], index: int
) -> Dict[str, Any]:
    return {name: values[index] for name, values in properties.items()}


def _add_to_exclude(excluded: Optional[Set[str]], to_exclude: str):
    if excluded is None:
        return {to_exclude}
    excluded.add(to_exclude)
    return excluded


def _check_property_name(
    properties: Dict[str, np.ndarray], property_name: str
):
    if property_name not in properties:
        raise ValueError(f'{property_name} is not in properties: {properties}')
