"""Contains napari color constants and utilities."""

import numpy as np

from napari.utils.colormaps.standardize_color import transform_color


class ColorValue(np.ndarray):
    """A 4x1 array that represents one RGBA color value."""

    @classmethod
    def __get_validators__(cls):
        yield cls.validate_type

    @classmethod
    def validate_type(cls, val):
        return transform_color(val)[0]


class ColorArray(np.ndarray):
    """An Nx4 array where each row of N represents one RGBA color value."""

    @classmethod
    def __get_validators__(cls):
        yield cls.validate_type

    @classmethod
    def validate_type(cls, val):
        return (
            np.empty((0, 4), np.float32)
            if len(val) == 0
            else transform_color(val)
        )
