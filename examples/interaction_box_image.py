"""
Demonstrate interaction box on image layer
"""

from skimage import data
import napari
from napari.utils.transforms import Affine

def on_transform_changed_start(event):
    # Save a copy of the intial transform
    viewer.layers.selection.active._start_affine = Affine(affine_matrix = viewer.layers.selection.active.affine.affine_matrix)

def on_transform_changed_drag(event):
    viewer.layers.selection.active.affine = event.value.compose(viewer.layers.selection.active._start_affine)
   
 
viewer = napari.view_image(data.astronaut(), rgb=True)
viewer.layers.selection.active.interactive = False

viewer.overlays.interaction_box.points = viewer.layers.selection.active.extent.world
viewer.overlays.interaction_box.show = True
viewer.overlays.interaction_box.show_vertices = True
viewer.overlays.interaction_box.show_handle = True
viewer.overlays.interaction_box.allow_new_selection = False

viewer.overlays.interaction_box.events.transform_start.connect(on_transform_changed_start)
viewer.overlays.interaction_box.events.transform_drag.connect(on_transform_changed_drag)

napari.run()
