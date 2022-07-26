# A simple driving example to test async slicing.

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

cells_data = cells3d()
membrane_data = np.squeeze(cells_data[:, 0, :, :])
delayed_membrane_data = DelayedArray(membrane_data, delay_s=1)

nuclei_data = np.squeeze(cells_data[:, 1, :, :])
delayed_nuclei_data = DelayedArray(nuclei_data, delay_s=1)

# Explicitly set multiscale because other guess_multiscale
# reads each slice with the delay.
viewer = napari.Viewer()
viewer.add_image(delayed_membrane_data, name='membrane', blending='additive', multiscale=False, colormap='magenta', contrast_limits=[1110, 23855])
viewer.add_image(delayed_nuclei_data, multiscale=False, name='nuclei', blending='additive', colormap='green', contrast_limits=[1600, 50000])

if __name__ == '__main__':
    napari.run()
