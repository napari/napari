"""
Multiple viewers
================

Create multiple viewers from the same script

.. tags:: gui
"""

from skimage import data

import napari

# add the image
photographer = data.camera()
viewer_a = napari.Viewer(title="viewer a")
layer_a = viewer_a.add_image(photographer, name='photographer')

# add the image in a new viewer window
astronaut = data.astronaut()
# Also view_path, add_shapes, add_points, add_labels etc.
viewer_b = napari.Viewer(title="viewer b")
layer_b = viewer_b.add_image(astronaut, name='astronaut')

if __name__ == '__main__':
    napari.run()
