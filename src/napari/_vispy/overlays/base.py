from typing import TYPE_CHECKING

from vispy.visuals.transforms import MatrixTransform, STTransform

from napari._vispy.utils.gl import BLENDING_MODES
from napari.components._viewer_constants import CanvasPosition
from napari.utils.events import disconnect_events
from napari.utils.translations import trans

if TYPE_CHECKING:
    from napari.layers import Layer


class VispyBaseOverlay:
    """
    Base overlay backend for vispy.

    Creates event connections between napari Overlay models and the
    vispy backend, translating them into rendering.
    """

    def __init__(self, *, overlay, node, parent=None) -> None:
        super().__init__()
        self.overlay = overlay

        self.node = node
        self.node.order = self.overlay.order

        self.overlay.events.visible.connect(self._on_visible_change)
        self.overlay.events.opacity.connect(self._on_opacity_change)
        self.overlay.events.blending.connect(self._on_blending_change)

        if parent is not None:
            self.node.parent = parent

    def _on_visible_change(self) -> None:
        self.node.visible = self.overlay.visible

    def _on_opacity_change(self) -> None:
        self.node.opacity = self.overlay.opacity

    def _on_blending_change(self) -> None:
        self.node.set_gl_state(**BLENDING_MODES[self.overlay.blending])
        self.node.update()

    def reset(self) -> None:
        self._on_visible_change()
        self._on_opacity_change()
        self._on_blending_change()

    def close(self) -> None:
        disconnect_events(self.overlay.events, self)
        self.node.transforms = MatrixTransform()
        self.node.parent = None


class VispyCanvasOverlay(VispyBaseOverlay):
    """
    Vispy overlay backend for overlays that live in canvas space.
    """

    def __init__(self, *, overlay, node, parent=None) -> None:
        super().__init__(overlay=overlay, node=node, parent=parent)

        # offsets and size are used to control fine positioning, and will depend
        # on the subclass and visual that needs to be rendered
        self.x_offset = 10.0
        self.y_offset = 10.0
        self.x_size = 0.0
        self.y_size = 0.0
        self.node.transform = STTransform()
        self.overlay.events.position.connect(self._on_position_change)

    def _on_position_change(self, event=None):
        # subclasses should set sizes correctly and adjust offsets to get
        # the optimal positioning
        if self.node.parent is None:
            return
        x_max, y_max = list(self.node.parent.size)
        position = self.overlay.position

        if position == CanvasPosition.TOP_LEFT:
            transform = [self.x_offset, self.y_offset, 0, 0]
        elif position == CanvasPosition.TOP_CENTER:
            transform = [x_max / 2 - self.x_size / 2, self.y_offset, 0, 0]
        elif position == CanvasPosition.TOP_RIGHT:
            transform = [
                x_max - self.x_size - self.x_offset,
                self.y_offset,
                0,
                0,
            ]
        elif position == CanvasPosition.BOTTOM_LEFT:
            transform = [
                self.x_offset,
                y_max - self.y_size - self.y_offset,
                0,
                0,
            ]
        elif position == CanvasPosition.BOTTOM_CENTER:
            transform = [
                x_max / 2 - self.x_size / 2,
                y_max - self.y_size - self.y_offset,
                0,
                0,
            ]
        elif position == CanvasPosition.BOTTOM_RIGHT:
            transform = [
                x_max - self.x_size - self.x_offset,
                y_max - self.y_size - self.y_offset,
                0,
                0,
            ]
        else:
            raise ValueError(
                trans._(
                    'Position {position} not recognized.',
                    deferred=True,
                    position=position,
                )
            )

        self.node.transform.translate = transform
        scale = abs(self.node.transform.scale[0])
        self.node.transform.scale = [scale, 1, 1, 1]

    def reset(self) -> None:
        super().reset()
        self._on_position_change()


class VispySceneOverlay(VispyBaseOverlay):
    """
    Vispy overlay backend for overlays that live in scene (2D or 3D) space.
    """

    def __init__(self, *, overlay, node, parent=None) -> None:
        super().__init__(overlay=overlay, node=node, parent=parent)
        self.node.transform = MatrixTransform()


class LayerOverlayMixin:
    def __init__(self, *, layer: 'Layer', overlay, node, parent=None) -> None:
        super().__init__(
            node=node,
            overlay=overlay,
            parent=parent,
        )
        self.layer = layer
        # need manual connection here because these overlays are not necessarily
        # always a child of the actual vispy node of the layer (eg, canvas overlays)
        self.layer.events.visible.connect(self._on_visible_change)

    def _on_visible_change(self) -> None:
        self.node.visible = self.overlay.visible and self.layer.visible

    def close(self) -> None:
        disconnect_events(self.layer.events, self)
        super().close()


class ViewerOverlayMixin:
    def __init__(self, *, viewer, overlay, node, parent=None) -> None:
        super().__init__(
            node=node,
            overlay=overlay,
            parent=parent,
        )
        self.viewer = viewer

    def close(self) -> None:
        disconnect_events(self.viewer.events, self)
        super().close()
