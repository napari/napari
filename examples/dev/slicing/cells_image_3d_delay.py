import numpy as np
import napari
from time import sleep
from skimage.data import cells3d

class DelayedArray:

    def __init__(self, array, *, delay_s: float):
        self.array = array
        self.delay_s = delay_s

    def __getitem__(self, key):
        sleep(self.delay_s)
        return self.array.__getitem__(key)

    @property
    def dtype(self):
        return self.array.dtype

    @property
    def shape(self):
        return self.array.shape

data = np.squeeze(cells3d()[:, 1, :, :])
delayed_data = DelayedArray(data, delay_s=1)

viewer = napari.view_image(delayed_data, multiscale=False, contrast_limits=[0, 32000])

if __name__ == '__main__':
    napari.run()
