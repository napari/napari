import warnings

from vispy.scene.visuals import Ellipse

from napari._vispy.overlays.base import ViewerOverlayMixin, VispyCanvasOverlay
from napari._vispy.utils.visual import overlay_to_visual
from napari.components.overlays import CanvasOverlay
from napari.utils.color import ColorValue


# the overlay model should inherit from either CanvasOverlay or SceneOverlay
# depending on if it needs to live in "screen space" or "scene space"
# (i.e: if it should be affected by camera, dims, ndisplay, ...)
class DotOverlay(CanvasOverlay):
    """
    Example overlay using a colored dot to show some state
    """
    color: ColorValue = (0, 1, 0, 1)
    size: int = 10


# the vispy overlay class should handle connecting the model to the vispy visual
# we use the ViewerOverlayMixin because this overlay is attached to the viewer,
# and not a specific layer
class VispyDotOverlay(ViewerOverlayMixin, VispyCanvasOverlay):
    # all arguments are keyword-only. viewer, overlay and parent should always be present.
    def __init__(self, *, viewer, overlay, parent=None):
        # the node argument for the base class is the vispy visual
        # note that the center is (0, 0), cause we handle the shift with transforms
        super().__init__(
            node=Ellipse(center=(0, 0)),
            viewer=viewer,
            overlay=overlay,
            parent=parent,
        )

        # we also need to connect events from the model to callbacks that update the visual
        self.overlay.events.color.connect(self._on_color_change)
        self.overlay.events.size.connect(self._on_size_change)
        # no need to connect position, since that's in the base classes of CanvasOverlay

        # at the end of the init of subclasses of VispyBaseOverlay we always
        # need to call reset to initialize properly
        self.reset()

    def _on_color_change(self, event=None):
        self.node.color = self.overlay.color

    def _on_position_change(self, event=None):
        # we can overload the position changing to account for the size, so that the dot
        # always sticks to the edge; there are `offset` attributes specifically for this
        self.x_offset = self.y_offset = self.overlay.size / 2
        super()._on_position_change()

    def _on_size_change(self, event=None):
        self.node.radius = self.overlay.size / 2
        # trigger position update since the radius changed
        self._on_position_change()

    # we should always add all the new callbacks to the reset() method
    def reset(self):
        super().reset()
        self._on_color_change()
        self._on_size_change()


# for napari to know how to use this overlay, we need to add it to the overlay_to_visual dict
# this will ideally be exposed at some point
overlay_to_visual[DotOverlay] = VispyDotOverlay


# note that we're accessing private attributes externally, which triggers a bunch of warnings.
# suppress them for the purpose of this example
if __name__ == '__main__':
    import napari

    viewer = napari.Viewer()
    # we also need to add at least a layer to see any overlay,
    # since the canvas is otherwise covered by the welcome widget
    viewer.add_shapes()

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        # add the overlay to the viewer (currently private attribute)
        viewer._overlays['dot'] = DotOverlay(visible=True)
        # there is currently no automation on adding a new overlay, so we also need to
        # manually trigger the generation of the visual
        viewer.window._qt_viewer._add_overlay(viewer._overlays['dot'])

    # let's make a simple widget to control the overlay
    from magicgui import magicgui

    from napari.components._viewer_constants import CanvasPosition

    @magicgui(
        auto_call=True,
        color={'choices': ['red', 'blue', 'green', 'magenta']},
        size={'widget_type': 'Slider', 'min': 1, 'max': 100}
    )
    def control_dot(viewer: napari.Viewer, color='red', size=20, position: CanvasPosition = 'top_left'):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            dot = viewer._overlays['dot']
            dot.color = color
            dot.size = size
            dot.position = position

    viewer.window.add_dock_widget(control_dot)
    control_dot()
