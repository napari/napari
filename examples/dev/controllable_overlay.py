# mypy: disable-error-code="attr-defined"
import warnings
from collections.abc import Generator
from copy import deepcopy
from typing import NamedTuple

import numpy as np
from magicgui import magicgui

import napari
from napari._vispy.mouse_event import NapariMouseEvent
from napari._vispy.overlays.interaction_box import VispySelectionBoxOverlay
from napari._vispy.utils.visual import overlay_to_visual
from napari.components.overlays import SelectionBoxOverlay
from napari.components.overlays.interaction_box import InteractionBoxHandle
from napari.layers import Image
from napari.layers.utils.interaction_box import (
    generate_interaction_box_vertices,
    get_nearby_handle,
)


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
    def _on_bounds_change(self) -> None:
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

# we add an image layer with random data;
data = np.random.randint(0, 255, size=(1024, 1024), dtype=np.uint8)
image = viewer.add_image(
    data,
    name='image',
)

# just for type checking
assert isinstance(image, Image)

# we recover the bounds of the image layer;
# this method will ensure that the overlay is drawn
# correctly in the viewer
layer_bounds = tuple(tuple(x.tolist()) for x in image._display_bounding_box_augmented([0, 1]).T)
image._overlays['selection_no_rotation'] = SelectionBoxNoRotation(
    bounds=layer_bounds, handles=True
)

# with some adjustments, the selection box overlay
# can be interacted with via mouse events;
# we need to first setup the mouse event handlers
# to allow for the interaction with the overlay;

# this is an helper named tuple to just
# allow us to access new mouse positions
# as x and y coordinates;
class MousePosition(NamedTuple):
    """Mouse position in data coordinates.

    An helper class to simplify access to
    the mouse position in data coordinates.

    Parameters
    ----------
    y : int
        Y coordinate of the mouse position.
    x : int
        X coordinate of the mouse position.
    """

    y: int
    x: int

    def __repr__(self) -> str:
        """Return a string representation of the mouse position."""
        return f"(y={self.y}, x={self.x})"

# this callback will handle the mouse events of
# dragging and dropping the selection box handles;
# we need to check if the mouse is within the bounds
# it will check if the mouse is in range of one of the
# overlay handles; if it is, we will set the selected handle
# to the handle that is closest to the mouse position;
# then we will set the bounds of the overlay to the
# new position of the mouse;
def resize_roi_box(
    layer: Image, event: "NapariMouseEvent"
) -> "Generator[None, None, None]":
    """Resize the ROI box based on mouse movement.

    Parameters
    ----------
    layer : DetectorLayer
        The layer to resize the ROI box for.
    event : NapariMouseEvent
        The event triggered by mouse movement.

    Yields
    ------
    None
        This is a generator function that handles mouse dragging.
    """
    if len(event.dims_displayed) != 2:
        return

    # Get the selected handle
    selected_handle = layer._overlays["selection_no_rotation"].selected_handle
    if selected_handle is None or selected_handle in [
        InteractionBoxHandle.INSIDE,
        InteractionBoxHandle.ROTATION,
    ]:
        # If no handle is selected or the selected handle is INSIDE or ROTATION, do nothing
        return

    current_bounds = deepcopy(layer._overlays["selection_no_rotation"].bounds)

    width, height = layer.data.shape[0], layer.data.shape[1]

    yield

    # Main event loop for handling drag events
    while event.type == "mouse_move":
        # this event.handled assignment should prevent propagation
        # to the pan-zoom event handler, but it does not and the
        # pan-zoom event handler is called anyway
        event.handled = True
        mouse_pos = MousePosition(
            *tuple(
                layer.world_to_data(event.position)[event.dims_displayed].astype(
                    int
                )
            )
        )

        # check if the mouse position is within the bounds of the image
        match selected_handle:
            case InteractionBoxHandle.CENTER_LEFT:
                if mouse_pos.x >= 0 and mouse_pos.x <= width - 1:
                    current_bounds = (
                        (mouse_pos.x, layer._overlays["selection_no_rotation"].bounds[0][1]),
                        layer._overlays["selection_no_rotation"].bounds[1],
                    )
            case InteractionBoxHandle.CENTER_RIGHT:
                if mouse_pos.x >= 1 and mouse_pos.x <= width:
                    current_bounds = (
                        layer._overlays["selection_no_rotation"].bounds[0],
                        (mouse_pos.x, layer._overlays["selection_no_rotation"].bounds[1][1]),
                    )
            case InteractionBoxHandle.TOP_LEFT:
                if (mouse_pos.y >= 0 and mouse_pos.y <= height - 1) and (
                    mouse_pos.x >= 0 and mouse_pos.x <= width - 1
                ):
                    print("top left not implemented", mouse_pos)
            case InteractionBoxHandle.TOP_RIGHT:
                if (mouse_pos.y >= 0 and mouse_pos.y <= height - 1) and (
                    mouse_pos.x >= 1 and mouse_pos.x <= width
                ):
                    print("top right not implemented", mouse_pos)
            case InteractionBoxHandle.TOP_CENTER:
                if mouse_pos.y >= 0 and mouse_pos.y <= height - 1:
                    current_bounds = (
                        (layer._overlays["selection_no_rotation"].bounds[0][0], mouse_pos.y),
                        layer._overlays["selection_no_rotation"].bounds[1],
                    )
            case InteractionBoxHandle.BOTTOM_CENTER:
                if mouse_pos.y >= 1 and mouse_pos.y <= height:
                    current_bounds = (
                        (layer._overlays["selection_no_rotation"].bounds[0][0], mouse_pos.y),
                        layer._overlays["selection_no_rotation"].bounds[1],
                    )
            case InteractionBoxHandle.BOTTOM_LEFT:
                if (mouse_pos.y >= 1 and mouse_pos.y <= height) and (
                    mouse_pos.x >= 0 and mouse_pos.x <= width - 1
                ):
                    print("bottom left not implemented", mouse_pos)
            case InteractionBoxHandle.BOTTOM_RIGHT:
                if (mouse_pos.y >= 1 and mouse_pos.y <= height) and (
                    mouse_pos.x >= 1 and mouse_pos.x <= width
                ):
                    print("bottom right not implemented", mouse_pos)

            case _:  # fallback on other handles, do nothing
                current_bounds = deepcopy(layer._overlays["selection_no_rotation"].bounds)

        layer._overlays["selection_no_rotation"].bounds = deepcopy(current_bounds)
        yield

# this callback will hightlight the overlay handles
# when the mouse hovers over them;
def highlight_roi_box_handles(layer: Image, event: NapariMouseEvent) -> None:
    """Highlight the hovered handle of a ROI.

    Parameters
    ----------
    layer : Image
        The layer to highlight the ROI box for.
    event : NapariMouseEvent
        The event triggered by mouse movement.
    """
    if len(event.dims_displayed) != 2:
        return

    # we work in data space so we're axis aligned which simplifies calculation
    # same as Layer.world_to_data
    world_to_data = (
        layer._transforms[1:].set_slice(layer._slice_input.displayed).inverse
    )
    pos = np.array(world_to_data(event.position))[event.dims_displayed]

    handle_coords = generate_interaction_box_vertices(
        *layer._overlays["selection_no_rotation"].bounds, handles=True
    )[:, ::-1]
    nearby_handle = get_nearby_handle(pos, handle_coords)
    if nearby_handle in [InteractionBoxHandle.INSIDE, InteractionBoxHandle.ROTATION]:
        return

    # set the selected vertex of the box to the nearby_handle (can also be INSIDE or None)
    layer._overlays["selection_no_rotation"].selected_handle = nearby_handle

# after defining the callbacks, we need to connect them to our layer;
# mouse_move_callbacks is a list of callbacks invoked when the mouse
# hovers over the layer;
# mouse_drag_callbacks is a list of callbacks invoked when the
# mouse is pressed, moved and released;
image.mouse_move_callbacks.append(highlight_roi_box_handles)
image.mouse_drag_callbacks.append(resize_roi_box)


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
