"""
Get a reference to the current napari viewer.
"""

import napari

# create viewer
viewer = napari.Viewer()

# lose reference to viewer
viewer = 'oops no viewer here'

# get that reference again
viewer = napari.current_viewer