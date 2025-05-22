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
# default behavior of the selection box; we redefine
# the handles attribute to True to show the handles
class SelectionBoxNoRotation(SelectionBoxOverlay):
    """Selection box overlay with no rotation handle."""


# we also need to define an equivalent vispy overlay;
# again, we subclass from VispySelectionBoxOverlay
class VispySelectionBoxNoRotation(VispySelectionBoxOverlay):
    """Vispy selection box overlay with no rotation handle."""

    def _on_bounds_change(self):
        if self.layer._slice_input.ndisplay == 2:
            top_left, bot_right = self.overlay.bounds
            self.node.set_data(
                # invert axes for vispy
                top_left[::-1],
                bot_right[::-1],
                handles=self.overlay.handles,
                selected=self.overlay.selected_handle,
                rotation=False,
            )


# register the new overlay classes
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


@magicgui
def toggle_overlay(
    viewer: napari.Viewer, transform_box: bool = False, selection_no_rotation: bool = False
) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        transform = viewer.layers['image']._overlays["transform_box"]
        selection = viewer.layers['image']._overlays['selection_no_rotation']
        transform.visible = transform_box
        selection.visible = selection_no_rotation


viewer.window.add_dock_widget(toggle_overlay)

if __name__ == '__main__':
    napari.run()
