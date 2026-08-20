import warnings

import numpy as np
from vispy.scene.visuals import Text

import napari
from napari._vispy.overlays.base import ViewerOverlayMixin, VispyCanvasOverlay
from napari._vispy.utils.visual import overlay_to_visual
from napari.components.overlays import CanvasOverlay


# the overlay model should inherit from either CanvasOverlay or SceneOverlay
# depending on whether it needs to live in "screen space" or "scene space"
# (i.e: if it should be affected by camera, dims, ndisplay, ...)
class OrientationOverlay(CanvasOverlay):
    """Orientation marker at one of the cardinal directions of the canvas."""
    text: str
    size: int = 10

# the vispy overlay class should handle connecting the model to the vispy
# visual we use the ViewerOverlayMixin because this overlay is attached to the
# viewer, and not a specific layer
class VispyOrientationOverlay(ViewerOverlayMixin, VispyCanvasOverlay):
    """Orientation marker at one of the cardinal directions of the canvas."""
    # all arguments are keyword-only. viewer, overlay and parent should always
    # be present.
    def __init__(self, **kwargs):
        # the node argument for the base class is the vispy visual
        super().__init__(
            node=Text(text='', bold=True, color='white', font_size=10),
            **kwargs
        )
        # we need to connect events from the model to callbacks that update
        # the visual
        self.overlay.events.text.connect(self._on_text_change)
        self.overlay.events.size.connect(self._on_size_change)

        # we *don't* need to connect position, because that's done for us in
        # the base classes of CanvasOverlay. We *can* overload
        # `_on_position_change` if we want to do some extra work.
        # `self.x_offset` and `self.y_offset` can be set in the overload to
        # nudge the canvas overlay around.

        # at the end of the init of subclasses of VispyBaseOverlay we always
        # need to call reset to initialize properly
        self.reset()

    def _on_text_change(self, event=None):
        self.node.text = self.overlay.text
        # trigger position update since the overall size of the visual *may*
        # change if the text changes
        self._on_position_change()

    def _on_size_change(self, event=None):
        self.node.font_size = self.overlay.size
        # trigger position update since the size changed
        self._on_position_change()

    # always add all new callbacks to the reset() method
    def reset(self):
        super().reset()
        self._on_text_change()
        self._on_size_change()


# for napari to know how to use this overlay, we need to add it to the
# overlay_to_visual dict. This will ideally be in a public API at some point
overlay_to_visual[OrientationOverlay] = VispyOrientationOverlay

viewer = napari.Viewer()
# we also need to add at least a layer to see any overlay,
# since the canvas is otherwise covered by the welcome widget
viewer.add_image(np.random.rand(10, 10))

# note that we're accessing private attributes externally, which triggers a
# bunch of warnings. We suppress them for the purpose of this example.
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    viewer.canvas.overlays.orientation_n = OrientationOverlay(
        visible=True, text='N', position='top_center'
    )
    viewer.canvas.overlays.orientation_s = OrientationOverlay(
        visible=True, text='S', position='bottom_center'
    )
    viewer.canvas.overlays.orientation_w = OrientationOverlay(
        visible=True, text='W', position='middle_left'
    )
    viewer.canvas.overlays.orientation_e = OrientationOverlay(
        visible=True, text='E', position='middle_right'
    )


if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        napari.run()
