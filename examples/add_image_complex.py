"""
Display one image using the add_image API.
"""

import napari

# from skimage import data
# from scipy.fftpack import fft2, fftshift

# im = data.astronaut().sum(-1)
# ny, nx = im.shape
# beta = 5
# win = np.stack(np.meshgrid(np.kaiser(ny, beta), np.kaiser(nx, beta))).prod(0)
# imf = np.log(fftshift(fft2(im * win)))


with napari.gui_qt():
    # create the viewer with an image
    viewer = napari.view_image(napari.utils.complex.complex_ramp())
