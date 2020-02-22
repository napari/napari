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


with napari.gui_qt():
    # create the viewer with an image
    viewer = napari.Viewer()
    viewer.add_image(im, name='astronaut')
    viewer.add_image(imf, name='FFT of astronaut')

    # a test sample in which the X dimension is a phase ramp from -ip to pi
    # and Y is a magnitude ramp from 0 - 10
    complex_ramp = napari.utils.complex.complex_ramp(512)
    viewer.add_image(complex_ramp, name="complex ramp")
