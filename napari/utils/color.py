"""Contains napari color constants and utilities."""

from types import GeneratorType
from typing import Union

import numpy as np

from napari.utils.colormaps.standardize_color import transform_color


class ColorValue(np.ndarray):
    """A custom pydantic field type for storing one color value."""

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(
        cls, value: Union[np.ndarray, list, tuple, str, None]
    ) -> np.ndarray:
        """Validates and coerces the given value into an array storing one color.

        Parameters
        ----------
        value
            A supported single color value, which must be one of the following.

            - A supported RGB(A) sequence of floating point values in [0, 1].
            - A CSS3 color name: https://www.w3.org/TR/css-color-3/#svg-color
            - A single character matplotlib color name: https://matplotlib.org/stable/tutorials/colors/colors.html#specifying-colors
            - An RGB(A) hex code string.

        Returns
        ----------
        np.ndarray
            An RGBA color vector of floating point values in [0, 1].
        """
        return transform_color(value)[0]


class ColorArray(np.ndarray):
    """A custom pydantic field type for storing multiple color values."""

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(
        cls, value: Union[np.ndarray, list, tuple, GeneratorType, None]
    ) -> np.ndarray:
        """Validates and coerces the given value into an array storing many colors.

        Parameters
        ----------
        value
            A supported sequence or generator of single color values.
            See ``ColorValue.validate`` for valid single color values.

        Returns
        ----------
        np.ndarray
            An array of N colors where each row is an RGBA color vector with
            floating point values in [0, 1].
        """
        return (
            np.empty((0, 4), np.float32)
            if len(value) == 0
            else transform_color(value)
        )
