"""Contains napari color constants and utilities."""

from typing import Any, Self, overload

import numpy as np
from pydantic import GetCoreSchemaHandler
from pydantic_core import CoreSchema, core_schema

from napari.utils.colormaps.standardize_color import transform_color

ColorValueParam = np.ndarray | list | tuple | str | None
ColorArrayParam = np.ndarray | list | tuple | str | None


class ColorValue(np.ndarray):
    """A custom pydantic field type for storing one color value.

    Using this as a field type in a pydantic model means that validation
    of that field (e.g. on initialization or setting) will automatically
    use the ``validate`` method to coerce a value to a single color.
    """

    def __new__(cls, value: ColorValueParam) -> Self:
        return cls.validate(value)

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        validate_schema = core_schema.no_info_plain_validator_function(
            cls.validate
        )

        serialize_schema = core_schema.plain_serializer_function_ser_schema(
            lambda x: x.tolist(),
            when_used='json',  # Only convert to list for JSON; keep as array for Python
        )

        return core_schema.json_or_python_schema(
            # Schema for JSON inputs (always run validation)
            json_schema=validate_schema,
            # Schema for Python inputs (allow existing instances to pass through)
            python_schema=core_schema.union_schema(
                [
                    core_schema.is_instance_schema(cls),
                    validate_schema,
                ],
                mode='left_to_right',
            ),
            serialization=serialize_schema,
        )

    @classmethod
    def validate(cls, value: ColorValueParam) -> Self:
        """Validates and coerces the given value into an array storing one color.

        Parameters
        ----------
        value : Union[np.ndarray, list, tuple, str, None]
            A supported single color value, which must be one of the following.

            - A supported RGB(A) sequence of floating point values in [0, 1].
            - A CSS3 color name: https://www.w3.org/TR/css-color-3/#svg-color
            - A single character matplotlib color name: https://matplotlib.org/stable/tutorials/colors/colors.html#specifying-colors
            - An RGB(A) hex code string.

        Returns
        -------
        np.ndarray
            An RGBA color vector of floating point values in [0, 1].

        Raises
        ------
        ValueError, AttributeError, KeyError
            If the value is not recognized as a color.

        Examples
        --------
        Coerce an RGBA array-like.

        >>> ColorValue.validate([1, 0, 0, 1])
        array([1., 0., 0., 1.], dtype=float32)

        Coerce a CSS3 color name.

        >>> ColorValue.validate('red')
        array([1., 0., 0., 1.], dtype=float32)

        Coerce a matplotlib single character color name.

        >>> ColorValue.validate('r')
        array([1., 0., 0., 1.], dtype=float32)

        Coerce an RGB hex-code.

        >>> ColorValue.validate('#ff0000')
        array([1., 0., 0., 1.], dtype=float32)
        """
        return transform_color(value)[0].view(cls)


class ColorArray(np.ndarray):
    """A custom pydantic field type for storing an array of color values.

    Using this as a field type in a pydantic model means that validation
    of that field (e.g. on initialization or setting) will automatically
    use the ``validate`` method to coerce a value to an array of colors.
    """

    def __new__(cls, value: ColorArrayParam) -> Self:
        return cls.validate(value)

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        validate_schema = core_schema.no_info_plain_validator_function(
            cls.validate
        )

        serialize_schema = core_schema.plain_serializer_function_ser_schema(
            lambda x: x.tolist(),
            when_used='json',  # Only convert to list for JSON; keep as array for Python
        )

        return core_schema.json_or_python_schema(
            # Schema for JSON inputs (always run validation)
            json_schema=validate_schema,
            # Schema for Python inputs (allow existing instances to pass through)
            python_schema=core_schema.union_schema(
                [
                    core_schema.is_instance_schema(cls),
                    validate_schema,
                ],
                mode='left_to_right',
            ),
            serialization=serialize_schema,
        )

    def __sizeof__(self) -> int:
        return super().__sizeof__() + self.nbytes

    @classmethod
    def validate(cls, value: ColorArrayParam) -> Self:
        """Validates and coerces the given value into an array storing many colors.

        Parameters
        ----------
        value : Union[np.ndarray, list, tuple, None]
            A supported sequence of single color values.
            See ``ColorValue.validate`` for valid single color values.
            In general each value should be of the same type, so avoid
            passing values like ``['red', [0, 0, 1]]``.

        Returns
        -------
        np.ndarray
            An array of N colors where each row is an RGBA color vector with
            floating point values in [0, 1].

        Raises
        ------
        ValueError, AttributeError, KeyError
            If the value is not recognized as an array of colors.

        Examples
        --------
        Coerce a list of CSS3 color names.

        >>> ColorArray.validate(['red', 'blue'])
        array([[1., 0., 0., 1.],
               [0., 0., 1., 1.]], dtype=float32)

        Coerce a tuple of matplotlib single character color names.

        >>> ColorArray.validate(('r', 'b'))
        array([[1., 0., 0., 1.],
               [0., 0., 1., 1.]], dtype=float32)
        """
        # Special case an empty supported sequence because transform_color
        # warns and returns an array containing a default color in that case.
        if isinstance(value, np.ndarray | list | tuple) and len(value) == 0:
            return np.empty((0, 4), np.float32).view(cls)
        return transform_color(value).view(cls)


# ITU-R BT.709 luma coefficients, shared by the luminance helpers below
_BT709_COEFFICIENTS = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


@overload
def rgb_to_luminance(rgb: ColorValue) -> float: ...


@overload
def rgb_to_luminance(
    rgb: ColorArray,
) -> np.ndarray[tuple[int], np.dtype[np.floating]]: ...


# can also work on more arbitrarily shaped arrays and with values outside of [0, 1]
@overload
def rgb_to_luminance(
    rgb: np.ndarray[tuple[int, ...]],
) -> np.ndarray[tuple[int, ...], np.dtype[np.floating]]: ...


def rgb_to_luminance(
    rgb: ColorValue | ColorArray | np.ndarray,
) -> (
    float
    | np.ndarray[tuple[int], np.dtype[np.floating]]
    | np.ndarray[tuple[int, ...], np.dtype[np.floating]]
):
    """Convert RGB(A) values to perceived luminance.

    Uses ITU-R BT.709 coefficients, compositing with the alpha channel
    if present.

    .. versionadded: 0.10.0
    """
    if rgb.shape[-1] == 3:
        return rgb @ _BT709_COEFFICIENTS
    if rgb.shape[-1] == 4:
        luminance = rgb[..., :3] @ _BT709_COEFFICIENTS
        # scale by alpha
        return luminance * rgb[..., 3]
    raise ValueError('can only convert rgb or rgba')


def _relative_luminance(rgb: ColorValue | np.ndarray) -> float:
    """Relative luminance of an sRGB color, as defined by WCAG 2.

    The sRGB channels are linearized (the 0.04045, 12.92 and 2.4 constants)
    and weighted with the ITU-R BT.709 coefficients. See
    https://www.w3.org/TR/WCAG21/#dfn-relative-luminance
    """
    rgb = np.asarray(rgb, dtype=float)[:3]
    rgb = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    return float(rgb @ _BT709_COEFFICIENTS)


def _contrast_ratio(
    color1: ColorValue | np.ndarray, color2: ColorValue | np.ndarray
) -> float:
    """WCAG 2 contrast ratio between two colors, from 1 to 21.

    See https://www.w3.org/TR/WCAG21/#dfn-contrast-ratio
    """
    lighter, darker = sorted(
        (_relative_luminance(color1), _relative_luminance(color2)),
        reverse=True,
    )
    return (lighter + 0.05) / (darker + 0.05)


def _contrasting_color(bgcolor: ColorValue) -> ColorValue:
    """Return a color that stands out against ``bgcolor``, keeping its alpha."""
    opposite = 1 - bgcolor
    # shift away from mid tones for better contrast
    opposite = 0.5 + (opposite - 0.5) * 1.2
    opposite = np.clip(opposite, 0, 1)
    # don't change alpha
    opposite[-1] = bgcolor[-1]
    return opposite.view(ColorValue)


def _readable_color(
    foreground_color: ColorValue | np.ndarray,
    background_color: ColorValue,
    min_contrast: float = 4.5,
) -> ColorValue:
    """Adjust ``foreground_color`` so it is readable on ``background_color``.

    The hue is kept: the color is blended toward black or white (whichever
    contrasts more with the background) until it reaches ``min_contrast``.
    The default 4.5 is the WCAG AA level for normal-size text, see
    https://www.w3.org/TR/WCAG21/#contrast-minimum
    """
    color = np.array([*foreground_color[:3], 1.0])
    target = max(
        (np.array([0.0, 0.0, 0.0, 1.0]), np.array([1.0, 1.0, 1.0, 1.0])),
        key=lambda c: _contrast_ratio(c, background_color),
    )
    for t in np.linspace(0, 1, 11):
        mixed = (1 - t) * color + t * target
        if _contrast_ratio(mixed, background_color) >= min_contrast:
            break
    return mixed.view(ColorValue)
