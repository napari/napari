from typing import Optional, Tuple
import numpy as np
from skimage.color import hsv2rgb
from ..utils.colormaps.colormaps import (
    vispy_or_mpl_colormap,
    AVAILABLE_COLORMAPS,
)


def complex_ramp(size=256, phase_range=(-np.pi, np.pi), mag_range=(0, 10)):
    """Returns a complex array where X ramps phase and Y ramps magnitude."""
    p0, p1 = phase_range
    phase_ramp = np.linspace(p0, p1 - 1 / size, size)
    m0, m1 = mag_range
    mag_ramp = np.linspace(m1, m0 + 1 / size, size)
    phase_ramp, mag_ramp = np.meshgrid(phase_ramp, mag_ramp)
    return mag_ramp * np.exp(1j * phase_ramp)


OptStr = Optional[str]
OptInt = Optional[int]


def complex2rgb(
    arr: np.ndarray,
    mapping: Tuple[OptStr, OptStr, OptStr] = ('phase', 'mag', None),
    scale: Tuple[OptInt, OptInt, OptInt] = (1, 1, 1),
    rmax: Optional[float] = None,
    phase_shift: float = np.pi,
) -> np.ndarray:
    """Convert complex array to RGB array.

    Mapping from phase or magnitude components to hue, saturation, or value is
    controlled by the ``mapping`` parameter.  See details below.

    Parameters
    ----------
    arr : np.ndarray
        Array with type np.complex or np.complex64
    mapping : list of str, optional
        mapping from Hue, Saturation, and Value to the corresponding complex
        component.  Only the first letter actually matters where 'phase' or 'p'
        maps to the phase component, and anything else ('mag' or 'abs') maps to
        the magnitude value.
        by default ['phase', 'mag', None]
    scale : list of int, optional
        Fraction of the HSV range occupied by the data range.
        by default [1, 1, 1]
    rmax : float, optional
        If provided clip the maximum mag components at `rmax`
        by default (None), the full range of data is used.
    phase_shift: float, optional
        amount to shift phase component on colorwheel.  by default np.pi.
        (a phase shift of np.pi works better if phase is encoding either
        saturation or value, and works fine for hue as well).

    Returns
    -------
    np.ndarray
        Array of shape arr.shape + (3,) with the RGB values.
    """
    assert len(mapping) == 3, 'Mapping must be a list or tuple with 3 items'

    def abs_func(_arr, _scale):
        absmax = rmax or np.abs(_arr).max()
        return np.clip(np.abs(_arr) / absmax, 0, 1) * _scale

    def ang_func(_arr, _scale):
        return (np.angle(_arr) + phase_shift) / (2 * np.pi) % _scale

    HSV = np.zeros(arr.shape + (3,), dtype='float')
    for i, component in enumerate(mapping):
        if component:
            func = ang_func if component.lower().startswith('p') else abs_func
            HSV[..., i] = func(arr, scale[i])
        else:
            HSV[..., i] = scale[i] if scale[i] is not None else 1

    return hsv2rgb(HSV)


def complex2colormap(
    arr: np.ndarray,
    colormap: str = 'twilight_shifted',
    rmax: Optional[float] = None,
    gamma: float = 0.8,
    phase_range: Tuple[float, float] = (-np.pi / 6, np.pi / 6),
) -> np.ndarray:
    """Convert a complex array to RGB array.

    Phase is mapped to color, while magnitude is mapped to brightness

    Parameters
    ----------
    arr : np.ndarray
        Array with type np.complex or np.complex64
    colormap : str, optional
        The colormap to use for phase info, by default 'twilight_shifted'
    rmax : float, optional
        The maximum magnitude to show, by default: ``np.abs(arr).max()``
    gamma : float, optional
        Controls the linearity of mapping between complex magnitue and
        brightness, by default 0.8
    phase_range : 2-tuple, optional
        set min/max phase range, by default (-np.pi / 6, np.pi / 6)

    Returns
    -------
    RGB : np.ndarray
        3-channel RGB numpy array
    """
    if colormap in AVAILABLE_COLORMAPS:
        cmap = AVAILABLE_COLORMAPS[colormap]
    else:
        cmap = vispy_or_mpl_colormap(colormap)

    # create RGB image from phase information
    p0, p1 = phase_range
    phase = (np.angle(arr) - p0) / (p1 - p0)
    RGB = cmap[phase.ravel()].RGB.reshape(phase.shape + (3,))

    # scale intensity of RGB image by the magnitude component
    absmax = rmax or np.abs(arr).max()
    intensity = np.clip(np.abs(arr) / absmax, 0, 1) ** gamma
    RGB = (RGB * np.expand_dims(intensity, 2)).astype(np.uint8)

    return RGB
