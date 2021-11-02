"""
Demonstrate interaction box on image layer
"""

from skimage import data
import numpy as np
import napari
from napari.utils.transforms import Affine

def on_transform_changed_start(event):
    # Save a copy of the intial transform
    viewer.layers.selection.active._start_affine = Affine(affine_matrix = viewer.layers.selection.active.affine.affine_matrix)

def on_transform_changed_drag(event):
    viewer.layers.selection.active.affine = event.value.compose(viewer.layers.selection.active._start_affine)
   
 
viewer = napari.view_image(data.astronaut(), rgb=True)
viewer.layers.selection.active.interactive = False

# All four corners are needed so that application of the transform after rotation results the right box
extent = viewer.layers.selection.active.extent.world
corners_in_world = np.array([
    [extent[0][0],extent[0][1]],
    [extent[1][0],extent[0][1]],
    [extent[1][0],extent[1][1]],
    [extent[0][0],extent[1][1]]
])
viewer.overlays.interaction_box.points = corners_in_world

viewer.overlays.interaction_box.show = True
viewer.overlays.interaction_box.show_vertices = True
viewer.overlays.interaction_box.show_handle = True
viewer.overlays.interaction_box.allow_new_selection = False

viewer.overlays.interaction_box.events.transform_start.connect(on_transform_changed_start)
viewer.overlays.interaction_box.events.transform_drag.connect(on_transform_changed_drag)

napari.run()
