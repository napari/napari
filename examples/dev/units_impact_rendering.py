import numpy as np

import napari

data = np.random.random(1000).reshape(10, 100)

viewer = napari.Viewer()
viewer.add_image(data, units=('mm', 'μm'), name='image1', scale=(0.01, 1))
viewer.add_image(data, units=('μm', 'μm'), name='image2', scale=(10, 1), colormap="red")

napari.run()
