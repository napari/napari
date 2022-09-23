"""
Custom image interpolation kernels
"""

from skimage import data
import napari
import numpy as np
from scipy.signal.windows import gaussian
from magicgui import magicgui

viewer = napari.view_image(data.astronaut(), rgb=True, interpolation='custom')


def gaussian_kernel(size, sigma):
    gauss = gaussian(size, sigma)
    return np.outer(gauss, gauss)


def sharpen_kernel():
    return np.array([
        [ 0, -1,  0],
        [-1,  5, -1],
        [ 0, -1,  0],
    ])


def ridge_detection_kernel():
    return np.array([
        [-1, -1, -1],
        [-1,  8, -1],
        [-1, -1, -1],
    ])


@magicgui(
    auto_call=True,
    size=dict(widget_type='Slider', min=1, max=20),
    sigma=dict(widget_type='FloatSlider', min=0.1, max=5, step=0.1),
    kernel_type=dict(choices=['gaussian', 'sharpen', 'ridge_detection']),
)
def gpu_kernel(image: napari.layers.Image, kernel_type: str = 'gaussian', size: int = 5, sigma: float = 1):
    if kernel_type == 'gaussian':
        image.custom_interpolation_kernel = gaussian_kernel(size, sigma)
    elif kernel_type == 'sharpen':
        image.custom_interpolation_kernel = sharpen_kernel()
    elif kernel_type == 'ridge_detection':
        image.custom_interpolation_kernel = ridge_detection_kernel()


viewer.window.add_dock_widget(gpu_kernel)
gpu_kernel()


if __name__ == '__main__':
    napari.run()
