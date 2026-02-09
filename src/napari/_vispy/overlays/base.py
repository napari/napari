from __future__ import annotations

from typing import TYPE_CHECKING

from vispy.scene.visuals import Rectangle
from vispy.visuals.transforms import MatrixTransform, STTransform

from napari._vispy.utils.gl import BLENDING_MODES
from napari.settings import get_settings
from napari.utils.color import ColorValue
from napari.utils.events import disconnect_events
from napari.utils.theme import get_theme

if TYPE_CHECKING:
    from napari.layers import Layer
    from napari.utils.events import Event


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

    NOTE: Subclasses must follow some rules:
    - ensure that when `_on_position_change` is called, the x_size and y_size
      attributes are already updated depending on the overlay size, to ensure
      proper tiling. Alternatively, override this method if the overlay is
      *not* supposed to be tiled
    - ensure that the napari Overlay model uses the `position` field correctly
      (must be a CanvasPosition enum if tileable, or anything else if "free")

    canvas_position_callback is set by the VispyCanvas object, and is responsible
    to update the position of all canvas overlays whenever necessary
    """

    def __init__(self, *, overlay, node, parent=None) -> None:
        super().__init__(overlay=overlay, node=node, parent=parent)
        self.x_size = 0.0
        self.y_size = 0.0
        self.node.transform = STTransform()
        self.overlay.events.position.connect(self._on_position_change)
        self.overlay.events.box.connect(self._on_box_change)
        self.overlay.events.box_color.connect(self._on_box_change)
        # ensure to connect last cause we need to know the primary node's visiblity
        self.overlay.events.visible.connect(
            self._on_box_change, position='last'
        )
        get_settings().appearance.events.theme.connect(self._on_box_change)
        self.canvas_position_callback = lambda: None

        self.box = Rectangle(center=(0, 0), border_width=0)

    def _on_box_change(self):
        if not self.overlay.box or not self.node.visible:
            self.box.parent = None
            return

        self.box.parent = self.node.parent

        # TODO: this should be related to tiling padding
        padding = 8
        self.box.width = self.x_size + padding
        self.box.height = self.y_size + padding
        self.box.center = self.x_size / 2, self.y_size / 2

        if self.overlay.box_color is None:
            if (
                self.node.parent is not None
                and self.node.parent.canvas.bgcolor
            ):
                background_color = ColorValue(
                    self.node.parent.canvas.bgcolor.rgba
                )
            else:
                background_color = ColorValue(
                    get_theme(
                        get_settings().appearance.theme
                    ).canvas.as_rgb_tuple()
                )
        else:
            background_color = self.overlay.box_color

        # make the color a bit transparent (copy to not override)
        background_color = background_color.copy()
        background_color[-1] *= 0.8

        self.box.color = background_color

        self.box.order = self.node.order - 1
        self.box.transform = self.node.transform

    def _on_blending_change(self) -> None:
        self.box.set_gl_state(**BLENDING_MODES[self.overlay.blending])
        super()._on_blending_change()

    def _on_position_change(self, event: Event | None = None) -> None:
        # NOTE: when subclasses call this method, they should first ensure sizes
        # (x_size, and y_size) are set correctly
        self._on_box_change()
        self.canvas_position_callback()

    def reset(self) -> None:
        super().reset()
        self._on_position_change()

    def close(self) -> None:
        super().close()
        self.canvas_position_callback = lambda: None


class VispySceneOverlay(VispyBaseOverlay):
    """
    Vispy overlay backend for overlays that live in scene (2D or 3D) space.
    """

    def __init__(self, *, overlay, node, parent=None) -> None:
        super().__init__(overlay=overlay, node=node, parent=parent)
        self.node.transform = MatrixTransform()


class LayerOverlayMixin:
    def __init__(self, *, layer: Layer, overlay, node, parent=None) -> None:
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
