from collections import OrderedDict
from enum import auto

from ...utils.misc import StringEnum
from ...utils.translations import trans


class Interpolation(StringEnum):
    """INTERPOLATION: Vispy interpolation mode.

    The spatial filters used for interpolation are from vispy's
    spatial filters. The filters are built in the file below:

    https://github.com/vispy/vispy/blob/main/vispy/glsl/build-spatial-filters.py
    """

    BESSEL = auto()
    BICUBIC = auto()
    LINEAR = auto()
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
    CUSTOM = auto()

    @classmethod
    def view_subset(cls):
        return (
            cls.BICUBIC,
            cls.LINEAR,
            cls.KAISER,
            cls.NEAREST,
            cls.SPLINE36,
        )


class Mode(StringEnum):
    """
    Mode: Interactive mode. The normal, default mode is PAN_ZOOM, which
    allows for normal interactivity with the canvas.

    TRANSFORM allows for manipulation of the layer transform.
    """

    TRANSFORM = auto()
    PAN_ZOOM = auto()


class ImageRendering(StringEnum):
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
    * average: average intensity projection. Cast a ray and display the
      average of values that were encountered.
    """

    TRANSLUCENT = auto()
    ADDITIVE = auto()
    ISO = auto()
    MIP = auto()
    MINIP = auto()
    ATTENUATED_MIP = auto()
    AVERAGE = auto()


class VolumeDepiction(StringEnum):
    """Depiction: 3D depiction mode for images.

    Selects a preset depiction mode in vispy
            * volume: images are rendered as 3D volumes.
            * plane: images are rendered as 2D planes embedded in 3D.
    """

    VOLUME = auto()
    PLANE = auto()


VOLUME_DEPICTION_TRANSLATION = OrderedDict(
    [
        (VolumeDepiction.VOLUME, trans._('volume')),
        (VolumeDepiction.PLANE, trans._('plane')),
    ]
)
