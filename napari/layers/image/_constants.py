from enum import auto, Enum
from ...utils.misc import StringEnum
import numpy as np
from functools import partial
from ...utils.complex import complex2rgb, complex2colormap


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


class Rendering(StringEnum):
    """Rendering: Rendering mode for the layer.

    Selects a preset rendering mode in vispy
            * translucent: voxel colors are blended along the view ray until
              the result is opaque.
            * mip: maxiumum intensity projection. Cast a ray and display the
              maximum value that was encountered.
            * attenuated_mip: attenuated maxiumum intensity projection. Cast a
              ray and attenuate values based on integral of encountered values,
              display the maximum value that was encountered after attenuation.
              This will make nearer objects appear more prominent.
            * additive: voxel colors are added along the view ray until
              the result is saturated.
            * iso: isosurface. Cast a ray until a certain threshold is
              encountered. At that location, lighning calculations are
              performed to give the visual appearance of a surface.
    """

    TRANSLUCENT = auto()
    ADDITIVE = auto()
    ISO = auto()
    MIP = auto()
    ATTENUATED_MIP = auto()


class ComplexRendering(Enum):
    """Mode for visualizing complex values

    * magnitude: uses np.abs
    * phase: uses np.angle
    * real: uses np.real
    * imaginary: uses np.imag
    """

    MAGNITUDE = partial(np.abs)
    PHASE = partial(np.angle)
    REAL = partial(np.real)
    IMAGINARY = partial(np.imag)
    COLORMAP = partial(complex2colormap)
    P2H_M2S = partial(complex2rgb, mapping=('p', 'm', None))
    P2H_M2V = partial(complex2rgb, mapping=('p', None, 'm'))
    P2H_M2SV = partial(complex2rgb, mapping=('p', 'm', 'v'))
    M2H_P2S = partial(complex2rgb, mapping=('m', 'p', None))
    M2H_P2V = partial(complex2rgb, mapping=('m', None, 'p'))
    M2H_P2SV = partial(complex2rgb, mapping=('m', 'p', 'p'))

    def __call__(self, *args, **kwargs):
        return self.value(*args, **kwargs)

    @classmethod
    def lower_members(cls):
        return list(map(str.lower, cls.__members__.keys()))

    @classmethod
    def rgb_members(cls):
        return [
            item
            for item in cls.__members__.values()
            if '2H_' in item.name or item.name == 'COLORMAP'
        ]
