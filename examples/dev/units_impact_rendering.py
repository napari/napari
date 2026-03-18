import skimage

import napari

data = skimage.data.cells3d()

ch1 = data[:, 0]
ch2 = data[:, 1]


viewer = napari.Viewer()
viewer.add_image(ch1, units=('nm', 'nm', 'nm'), name='ch1', scale=(210, 70, 70), colormap="magenta")
viewer.add_image(ch2, units=('μm', 'μm', 'μm'), name='ch2', scale=(0.210, 0.07, 0.07), colormap="green", blending='additive')

napari.run()
