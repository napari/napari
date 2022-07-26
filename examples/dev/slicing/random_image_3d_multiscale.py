import numpy as np
import napari
from skimage.transform import rescale

hires_data = np.random.rand(100, 1024, 1024)
midres_data = rescale(hires_data, 0.5, channel_axis=[0])
lowres_data = rescale(midres_data, 0.5, channel_axis=[0])
multiscale = [hires_data, midres_data, lowres_data]

viewer = napari.view_image(multiscale, multiscale=True)

if __name__ == '__main__':
    napari.run()
