import numpy as np
import napari
from skimage.data import cells3d

data = np.squeeze(cells3d()[:, 1, :, :])
viewer = napari.view_image(data, contrast_limits=[0, 32000])

if __name__ == '__main__':
    napari.run()
