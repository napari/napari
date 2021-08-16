import napari
import numpy as np

data_0 = np.zeros((20, 50, 50))
data_0[10:20, 25:35, 25:35] = 1

data_1 = np.zeros((50, 50, 50))
data_1[30:40, 25:35, 25:35] = 1

viewer = napari.view_image(data_0, colormap='magenta', rendering='iso')
viewer.add_image(data_1, colormap='green', rendering='iso')
viewer.add_points([[25, 30, 30]], size=4)

viewer.dims.ndisplay = 3

napari.run()