from enum import auto
from typing import Tuple

from ...utils.misc import StringEnum


class Interpolation(StringEnum):
    """INTERPOLATION: Vispy interpolation mode.

    The spatial filters used for interpolation are from vispy's
    spatial filters. The filters are built in the file below:

    https://github.com/vispy/vispy/blob/master/vispy/glsl/build-spatial-filters.py
    """

    BESSEL = auto()
    BICUBIC = auto()
    BILINEAR = auto()
    BLACKMAN = auto()
    CATROM = auto()
    GAUSSIAN = auto()
    HAMMING = auto()
    HANNING = auto()
    HERMITE = auto()
    KAISER = auto()
    LANCZOS = auto()
    MITCHELL = auto()
    NEAREST = auto()
    SPLINE16 = auto()
    SPLINE36 = auto()

    @classmethod
    def view_subset(cls):
        return (
            cls.BICUBIC,
            cls.BILINEAR,
            cls.KAISER,
            cls.NEAREST,
            cls.SPLINE36,
        )


class Interpolation3D(StringEnum):
    """INTERPOLATION: Vispy interpolation mode for volume rendering."""

    LINEAR = auto()
    NEAREST = auto()


class Rendering(StringEnum):
    """Rendering: Rendering mode for the layer.

    Selects a preset rendering mode in vispy
            * translucent: voxel colors are blended along the view ray until
              the result is opaque.
            * mip: maximum intensity projection. Cast a ray and display the
              maximum value that was encountered.
            * minip: minimum intensity projection. Cast a ray and display the
              minimum value that was encountered.
            * attenuated_mip: attenuated maximum intensity projection. Cast a
              ray and attenuate values based on integral of encountered values,
              display the maximum value that was encountered after attenuation.
              This will make nearer objects appear more prominent.
            * additive: voxel colors are added along the view ray until
              the result is saturated.
            * iso: isosurface. Cast a ray until a certain threshold is
              encountered. At that location, lighning calculations are
              performed to give the visual appearance of a surface.
            * iso_categorical: isosurface for categorical data (e.g., labels).
              Cast a ray until a non-background value is encountered. At that
              location, lighning calculations are performed to give the visual
              appearance of a surface.
            * average: average intensity projection. Cast a ray and display the
              average of values that were encountered.
    """

    TRANSLUCENT = auto()
    ADDITIVE = auto()
    ISO = auto()
    ISO_CATEGORICAL = auto()
    MIP = auto()
    MINIP = auto()
    ATTENUATED_MIP = auto()
    AVERAGE = auto()

    @classmethod
    def image_layer_subset(cls) -> Tuple['Rendering']:
        return (
            cls.TRANSLUCENT,
            cls.ADDITIVE,
            cls.ISO,
            cls.MIP,
            cls.MINIP,
            cls.ATTENUATED_MIP,
            cls.AVERAGE,
        )

    @classmethod
    def labels_layer_subset(cls) -> Tuple['Rendering']:
        return (cls.TRANSLUCENT, cls.ISO_CATEGORICAL)
