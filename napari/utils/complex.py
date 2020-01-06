import numpy as np
from skimage.color import hsv2rgb
from functools import partial


def complex_ramp(size=256, phase_range=(-np.pi, np.pi), mag_range=(0, 10)):
    """Returns a complex array where X ramps phase and Y ramps magnitude."""
    size = 256
    p0, p1 = phase_range
    phase_ramp = np.linspace(p0, p1 - 1 / size, size)
    m0, m1 = mag_range
    mag_ramp = np.linspace(m1, m0 + 1 / size, size)
    phase_ramp, mag_ramp = np.meshgrid(phase_ramp, mag_ramp)
    return mag_ramp * np.exp(1j * phase_ramp)


def complex2rgb(
    arr,
    mapping=['phase', 'mag', None],
    scale=[1, 1, 1],
    rmax=None,
    phase_shift=np.pi,
):
    """Convert complex array to RGB array.

    Parameters
    ----------
    arr : np.ndarray
        Array with type np.complex or np.complex64
    mapping : list, optional
        mapping from Hue, Saturation, and Value to the corresponding complex
        component.  Only the first letter actually matters where 'phase' or 'p'
        maps to the phase component, and anything else ('mag' or 'abs') maps to
        the magnitude value.
        by default ['phase', 'mag', None]
    scale : list, optional
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


pm_hs1_2rgb = partial(complex2rgb, mapping=['p', 'm', None])
