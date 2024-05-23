
import numpy as np

import napari

# Set the number of steps
num_steps = 2**17

base = np.linspace(start=1, stop=num_steps, num=num_steps).astype('uint32')
label_img = np.repeat(
        base.reshape([1, base.shape[0]]), int(num_steps/1000), axis=0
        )

viewer = napari.Viewer()
viewer.add_image(
        label_img,
        scale=(100, 1),
        colormap='viridis',
        contrast_limits=(0, num_steps),
        )

if __name__ == '__main__':
    napari.run()
