import warnings

import numpy as np
from magicgui import magicgui

import napari
from napari._vispy.overlays.base import LayerOverlayMixin, VispySceneOverlay
from napari._vispy.overlays.interaction_box import (
    InteractionBox,
)
from napari._vispy.utils.visual import overlay_to_visual
from napari.components.overlays import SceneOverlay
from napari.layers import Image
from napari.layers.utils.interaction_box import (
    InteractionBoxHandle,
    calculate_bounds_from_contained_points,
)


# define a model for the selection box overlay;
# we subclass from SelectionBoxOverlay to get the
# default behavior of the selection box; we redefine
# the handles attribute to True to show the handles
class SelectionBoxNoRotation(SceneOverlay):
    """Selection box overlay with no rotation handle."""

    rotation: bool = False
    handles: bool = True
    bounds: tuple[tuple[float, float], tuple[float, float]] = ((0, 0), (0, 0))
    selected_handle: InteractionBoxHandle | None = None

    def update_from_points(self, points: np.ndarray) -> None:
        """Create as a bounding box of the given points"""
        self.bounds = calculate_bounds_from_contained_points(points)


# we also need to define an equivalent vispy overlay;
# again, we subclass from VispySelectionBoxOverlay
class VispySelectionBoxNoRotation(LayerOverlayMixin, VispySceneOverlay):
    """Vispy selection box overlay with no rotation handle."""

    layer: Image
    node: InteractionBox
    overlay: SelectionBoxNoRotation

    def __init__(self, *, layer, overlay, parent=None) -> None:
        super().__init__(
            node=InteractionBox(),
            layer=layer,
            overlay=overlay,
            parent=parent,
        )
        self.overlay.events.bounds.connect(self._on_bounds_change)
        self.overlay.events.handles.connect(self._on_bounds_change)
        self.overlay.events.selected_handle.connect(self._on_bounds_change)
        self.overlay.events.visible.connect(self._on_visible_change)
        self.layer.events.set_data.connect(self._on_visible_change)

    def _on_bounds_change(self) -> None:
        if self.layer._slice_input.ndisplay == 2:
            top_left, bot_right = self.overlay.bounds
            self.node.set_data(
                # invert axes for vispy
                top_left[::-1],
                bot_right[::-1],
                handles=self.overlay.handles,
                selected=self.overlay.selected_handle,
                rotation=self.overlay.rotation,
            )

    def _on_visible_change(self) -> None:
        if self.layer._slice_input.ndisplay == 2:
            super()._on_visible_change()
            self._on_bounds_change()
        else:
            self.node.visible = False

    def reset(self):
        super().reset()
        self._on_bounds_change()


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
    bounds=[(0, 0), data.shape]
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
