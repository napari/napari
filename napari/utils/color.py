"""Contains napari color constants and utilities."""

import numpy as np

from napari.utils.colormaps.standardize_color import transform_color


class ColorValue(np.ndarray):
    """An array of shape (4,) that represents one RGBA color value."""

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, value):
        """Validates and coerces a value into a ``ColorValue``."""
        return transform_color(value)[0]


class ColorArray(np.ndarray):
    """An array of shape (N, 4) where each row of N represents one RGBA color value."""

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, value):
        """Validates and coerces a value into a ``ColorArray``."""
        return (
            np.empty((0, 4), np.float32)
            if len(value) == 0
            else transform_color(value)
        )
