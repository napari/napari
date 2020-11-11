"""
Display one image using the add_image API.
"""

import numpy as np
from scipy.fftpack import fftn, fftshift, ifftshift
from skimage import io

import napari

PSF = io.imread(
    'https://nic.med.harvard.edu/wp-content/uploads/2020/02/psf.tif'
)
nz, ny, nx = PSF.shape
beta = 3
win = np.stack(
    np.meshgrid(np.kaiser(nz, beta), np.kaiser(ny, beta), np.kaiser(nx, beta))
).prod(0)
OTF = np.log(fftshift(fftn(ifftshift(PSF * win))))

with napari.gui_qt():
    # create the viewer with an image
    viewer = napari.Viewer()
    viewer.add_image(PSF)
    viewer.add_image(OTF)
