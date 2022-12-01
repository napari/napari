from vispy.visuals.transforms import MatrixTransform, STTransform

from napari.components._viewer_constants import CanvasPosition
from napari.utils.events import disconnect_events
from napari.utils.translations import trans


class VispyBaseOverlay:
    def __init__(self, *, overlay, node, parent=None):
        super().__init__()
        self.overlay = overlay

        self.node = node
        self.node.order = self.overlay.order

        self.overlay.events.visible.connect(self._on_visible_change)
        self.overlay.events.opacity.connect(self._on_opacity_change)

        self.node.parent = parent

    def _on_visible_change(self):
        self.node.visible = self.overlay.visible

    def _on_opacity_change(self):
        self.node.opacity = self.overlay.opacity

    def reset(self):
        self._on_visible_change()
        self._on_opacity_change()

    def close(self):
        disconnect_events(self.overlay.events, self)
        self.node.transforms = MatrixTransform()
        self.node.parent = None


class VispyCanvasOverlay(VispyBaseOverlay):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.x_offset = 10
        self.y_offset = 10
        self.x_size = 0
        self.y_size = 0
        self.node.transform = STTransform()
        self.overlay.events.position.connect(self._on_position_change)
        self.node.events.parent_change.connect(self._on_parent_change)

    def _on_parent_change(self, event):
        if event.old is not None:
            disconnect_events(self, event.old.canvas)
        if event.new is not None and self.node.canvas is not None:
            event.new.canvas.events.resize.connect(self._on_position_change)

    def _on_position_change(self, event=None):
        if self.node.canvas is None:
            return
        x_max, y_max = list(self.node.canvas.size)
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

    def reset(self):
        super().reset()
        self._on_position_change()


class VispySceneOverlay(VispyBaseOverlay):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.node.transform = MatrixTransform()


class LayerOverlayMixin:
    def __init__(self, *, layer, **kwargs):
        super().__init__(**kwargs)
        self.layer = layer
        self.layer._overlays.events.removing.connect(self._close_if_removed)

    def _close_if_removed(self, event):
        if self.layer._overlays[event.key] is self:
            self.close()

    def close(self):
        disconnect_events(self.layer.events, self)
        super().close()


class ViewerOverlayMixin:
    def __init__(self, *, viewer, **kwargs):
        super().__init__(**kwargs)
        self.viewer = viewer

    def close(self):
        disconnect_events(self.viewer.events, self)
        super().close()
