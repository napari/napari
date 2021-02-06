import napari
from napari.layers.utils import experimental_link_layers
import numpy as np


viewer = napari.view_image(np.random.rand(3, 64, 64), channel_axis=0)

# link contrast_limits and gamma between all layers in viewer
# NOTE: you may also omit the second argument to link ALL valid, common
# attributes for the set of layers provided
experimental_link_layers(viewer.layers, ('contrast_limits', 'gamma'))

napari.run()
