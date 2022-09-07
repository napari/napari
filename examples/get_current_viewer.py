"""
Get current viewer
==================

Get a reference to the current napari viewer.

Whilst this example is contrived, it can be useful to get a reference to the
viewer when the viewer is out of scope.

.. tags:: gui
"""

import napari

# create viewer
viewer = napari.Viewer()

# lose reference to viewer
viewer = 'oops no viewer here'

# get that reference again
viewer = napari.current_viewer()
