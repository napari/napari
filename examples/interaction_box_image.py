"""
Demonstrate interaction box on image layer
"""

from skimage import data
import numpy as np
import napari
from napari.utils.transforms import Affine


def on_transform_changed_drag(event):
    viewer.layers.selection.active.affine = event.value
   
 
viewer = napari.view_image(data.astronaut(), rgb=True)
viewer.layers.selection.active.interactive = False


viewer.overlays.interaction_box.points = viewer.layers.selection.active.extent.world

viewer.overlays.interaction_box.show = True
viewer.overlays.interaction_box.show_vertices = True
viewer.overlays.interaction_box.show_handle = True
viewer.overlays.interaction_box.allow_new_selection = False

viewer.overlays.interaction_box.events.transform_drag.connect(on_transform_changed_drag)

napari.run()
