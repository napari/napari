import warnings

import numpy as np
from magicgui import magicgui

import napari
from napari._vispy.overlays.interaction_box import VispySelectionBoxOverlay
from napari._vispy.utils.visual import overlay_to_visual
from napari.components.overlays import SelectionBoxOverlay
from napari.layers import Image


# define a model for the selection box overlay;
# we subclass from SelectionBoxOverlay to get the
# default behavior of the selection box;
class SelectionBoxNoRotation(SelectionBoxOverlay):
    """Selection box overlay with no rotation handle."""


# we also need to define an equivalent vispy overlay;
# again, we subclass from VispySelectionBoxOverlay
class VispySelectionBoxNoRotation(VispySelectionBoxOverlay):
    """Vispy selection box overlay with no rotation handle."""

    # the _on_bounds_change method is the same as in the
    # original VispySelectionBoxOverlay, but we set
    # rotation to False to not draw the rotation handle
    def _on_bounds_change(self):
        if self.layer._slice_input.ndisplay == 2:
            top_left, bot_right = self.overlay.bounds
            self.node.set_data(
                # invert axes for vispy
                top_left[::-1],
                bot_right[::-1],
                handles=self.overlay.handles,
                selected=self.overlay.selected_handle,
                # by setting rotation to False,
                # the circle handle will not be drawn
                rotation=False,
            )


# before we can use the new overlay, we have to update
# the overlay_to_visual mapping to include our new overlay;
# this is necessary so that the correct vispy overlay
# is used when the overlay is created
overlay_to_visual[SelectionBoxNoRotation] = VispySelectionBoxNoRotation

viewer = napari.Viewer()

data = np.random.randint(0, 255, size=(1024, 1024), dtype=np.uint8)
image = viewer.add_image(
    data,
    name='image',
)

# just for type checking
assert isinstance(image, Image)

image._overlays['selection_no_rotation'] = SelectionBoxNoRotation(
    bounds=[(0, 0), data.shape], handles=True
)


# we use a simple magicgui widget to allow
# the toggling of the selection box overlay
# as demonstration
@magicgui
def toggle_overlay(
    viewer: napari.Viewer, toggle_selection_box: bool = False
) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        viewer.layers['image']._overlays['selection_no_rotation'].visible = toggle_selection_box

# add the widget to the viewer
viewer.window.add_dock_widget(toggle_overlay)

if __name__ == '__main__':
    napari.run()
