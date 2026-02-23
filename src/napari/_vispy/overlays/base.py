from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from vispy.scene.visuals import Rectangle
from vispy.visuals.transforms import MatrixTransform, STTransform

from napari._vispy.utils.gl import BLENDING_MODES
from napari.settings import get_settings
from napari.utils.color import ColorValue
from napari.utils.events import disconnect_events
from napari.utils.theme import get_theme

if TYPE_CHECKING:
    from napari.components.overlays import CanvasOverlay, Overlay, SceneOverlay
    from napari.components.viewer_model import ViewerModel
    from napari.layers import Layer
    from napari.utils.events import Event


class VispyBaseOverlay:
    """
    Base overlay backend for vispy.

    Creates event connections between napari Overlay models and the
    vispy backend, translating them into rendering.
    """

    overlay: Overlay
    viewer: ViewerModel

    def __init__(self, *, overlay, viewer, node, parent=None) -> None:
        super().__init__()
        self.overlay = overlay
        self.viewer = viewer

        self.node = node
        self.node.order = self.overlay.order

        self.overlay.events.visible.connect(self._on_visible_change)
        self.overlay.events.opacity.connect(self._on_opacity_change)
        self.overlay.events.blending.connect(self._on_blending_change)

        if parent is not None:
            self.node.parent = parent

    def _should_be_visible(self) -> bool:
        return self.overlay.visible

    def _on_visible_change(self) -> None:
        self.node.visible = self._should_be_visible()

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
        self.overlay.events.visible.disconnect(self._on_visible_change)
        self.overlay.events.opacity.disconnect(self._on_opacity_change)
        self.overlay.events.blending.disconnect(self._on_blending_change)
        disconnect_events(self.viewer.events, self)
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

    overlay: CanvasOverlay

    def __init__(self, *, overlay, viewer, node, parent=None) -> None:

        super().__init__(
            overlay=overlay, viewer=viewer, node=node, parent=parent
        )
        self.x_size = 0.0
        self.y_size = 0.0
        self.node.transform = STTransform()
        self.overlay.events.position.connect(self._on_position_change)
        self.overlay.events.box.connect(self._on_box_change)
        self.overlay.events.box_color.connect(self._on_box_change)
        get_settings().appearance.events.theme.connect(self._on_box_change)
        self.viewer.events.theme.connect(self._on_box_change)
        self.canvas_position_callback = lambda: None

        self.box = Rectangle(center=(0, 0), border_width=0)

    def _on_visible_change(self) -> None:
        super()._on_visible_change()
        self._on_box_change()

    def _on_box_change(self) -> None:
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
            bgcolor = self._get_canvas_bgcolor()
            # make the color a bit transparent
            bgcolor[-1] *= 0.8
        else:
            bgcolor = self.overlay.box_color

        self.box.color = bgcolor

        self.box.order = self.node.order - 1
        self.box.transform = self.node.transform

    def _get_canvas_bgcolor(self) -> ColorValue:
        if self.node.parent is not None and self.node.parent.canvas.bgcolor:
            return ColorValue(self.node.parent.canvas.bgcolor.rgba)

        return ColorValue(
            get_theme(get_settings().appearance.theme).canvas.as_rgb_tuple()
        )

    def _get_fgcolor(self) -> ColorValue:
        if not self.overlay.box or self.overlay.box_color is None:
            bgcolor = self._get_canvas_bgcolor()
        else:
            bgcolor = self.overlay.box_color
        return self._contrasting_color(bgcolor)

    def _contrasting_color(self, bgcolor: ColorValue) -> ColorValue:
        opposite = 1 - bgcolor
        # shift away from mid tones for better contrast
        opposite = 0.5 + (opposite - 0.5) * 1.2
        opposite = np.clip(opposite, 0, 1)
        # don't change alpha
        opposite[-1] = bgcolor[-1]
        return opposite

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
        self.box.parent = None
        self.canvas_position_callback = lambda: None


class VispySceneOverlay(VispyBaseOverlay):
    """
    Vispy overlay backend for overlays that live in scene (2D or 3D) space.
    """

    overlay: SceneOverlay

    def __init__(self, *, overlay, viewer, node, parent=None) -> None:
        super().__init__(
            overlay=overlay, viewer=viewer, node=node, parent=parent
        )
        self.node.transform = MatrixTransform()


class LayerOverlayMixin:
    layer: Layer

    def __init__(
        self, *, overlay, layer: Layer, viewer, node, parent=None
    ) -> None:
        super().__init__(
            node=node,
            overlay=overlay,
            viewer=viewer,
            parent=parent,
        )
        self.layer = layer
        # need manual connection here because these overlays are not necessarily
        # always a child of the actual vispy node of the layer (eg, canvas overlays)
        self.layer.events.visible.connect(self._on_visible_change)

    def _should_be_visible(self) -> bool:
        return self.overlay.visible and self.layer.visible

    def close(self) -> None:
        disconnect_events(self.layer.events, self)
        super().close()


class ViewerOverlayMixin:
    pass
