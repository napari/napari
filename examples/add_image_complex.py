"""
Display one image using the add_image API.
"""

import napari
import numpy as np
from skimage import data
from scipy.fftpack import fft2, fftshift, ifftshift

im = data.astronaut().sum(-1)
ny, nx = im.shape
beta = 5
win = np.stack(np.meshgrid(np.kaiser(ny, beta), np.kaiser(nx, beta))).prod(0)
imf = np.log(fftshift(fft2(ifftshift(im * win))))


def complex_ramp(size=256, phase_range=(-np.pi, np.pi), mag_range=(0, 10)):
    """Returns a complex array where X ramps phase and Y ramps magnitude."""
    p0, p1 = phase_range
    phase_ramp = np.linspace(p0, p1 - 1 / size, size)
    m0, m1 = mag_range
    mag_ramp = np.linspace(m1, m0 + 1 / size, size)
    phase_ramp, mag_ramp = np.meshgrid(phase_ramp, mag_ramp)
    return mag_ramp * np.exp(1j * phase_ramp)


with napari.gui_qt():
    # create the viewer with an image
    viewer = napari.Viewer()
    # a test sample in which the X dimension is a phase ramp from -ip to pi
    # and Y is a magnitude ramp from 0 - 10
    viewer.add_image(complex_ramp(512), name="complex ramp")
    viewer.add_image(imf, name='FFT of astronaut')
