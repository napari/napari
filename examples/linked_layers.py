"""Demonstrates the `link_layers` function.

This function takes a list of layers and an optional list of attributes, and
links them such that when one of the linked attributes changes on any of the
linked layers, all of the other layers follow.
"""
import napari
from napari.experimental import link_layers
import numpy as np


viewer = napari.view_image(np.random.rand(3, 64, 64), channel_axis=0)

# link contrast_limits and gamma between all layers in viewer
# NOTE: you may also omit the second argument to link ALL valid, common
# attributes for the set of layers provided
link_layers(viewer.layers, ('contrast_limits', 'gamma'))

# unlinking may be done with napari.experimental.unlink_layers

# this may also be done in a context manager:
# with napari.experimental.layers_linked([layers]):
#     ...


napari.run()
