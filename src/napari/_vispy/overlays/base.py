from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from vispy.scene.visuals import Rectangle
from vispy.visuals.transforms import MatrixTransform, STTransform

from napari._vispy.utils.gl import BLENDING_MODES
from napari.utils.color import ColorValue
from napari.utils.events import disconnect_events

if TYPE_CHECKING:
    from vispy.scene import Node, ViewBox
    from vispy.visuals.text.text import FontManager

    from napari.components.canvas import Canvas
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

    def __init__(
        self,
        *,
        overlay: Overlay,
        viewer: ViewerModel,
        node: Node,
        parent: ViewBox | None = None,
        font_manager: FontManager | None = None,
        font_family: str = 'OpenSans',
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.overlay = overlay
        self.viewer = viewer
        self.font_manager = font_manager
        self.font_family = font_family

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
    canvas: Canvas

    def __init__(
        self, *, overlay, canvas, viewer, node, parent=None, **kwargs
    ) -> None:

        super().__init__(
            overlay=overlay,
            canvas=canvas,
            viewer=viewer,
            node=node,
            parent=parent,
            **kwargs,
        )
        self.canvas = canvas

        self.x_size = 0.0
        self.y_size = 0.0
        self.node.transform = STTransform()
        self.overlay.events.position.connect(self._on_position_change)
        self.overlay.events.box.connect(self._on_box_change)
        self.overlay.events.box_color.connect(self._on_box_change)

        self.canvas.events.background_color.connect(self._on_box_change)

        self.box = Rectangle(center=(0, 0), border_width=0)

    def _on_visible_change(self) -> None:
        super()._on_visible_change()
        self._on_box_change()

    def _on_box_change(self) -> None:
        if not self.overlay.box or not self._should_be_visible():
            self.box.parent = None
            return

        self.box.parent = self.node.parent

        # TODO: this should be related to tiling padding
        padding = 8
        self.box.width = self.x_size + padding
        self.box.height = self.y_size + padding
        self.box.center = self.x_size / 2, self.y_size / 2

        if self.overlay.box_color is None:
            bgcolor = self.canvas.background_color
            # make the color a bit transparent
            bgcolor[-1] *= 0.8
        else:
            bgcolor = self.overlay.box_color

        self.box.color = bgcolor

        self.box.order = self.node.order - 1
        self.box.transform = self.node.transform

    def _get_fgcolor(self) -> ColorValue:
        if not self.overlay.box or self.overlay.box_color is None:
            bgcolor = self.canvas.background_color
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
        self.canvas.events._overlay_positions_changed()

    def reset(self) -> None:
        super().reset()
        self._on_position_change()

    def close(self) -> None:
        super().close()
        self.box.parent = None


class VispySceneOverlay(VispyBaseOverlay):
    """
    Vispy overlay backend for overlays that live in scene (2D or 3D) space.
    """

    overlay: SceneOverlay

    def __init__(
        self, *, overlay, viewer, node, parent=None, **kwargs
    ) -> None:
        super().__init__(
            overlay=overlay, viewer=viewer, node=node, parent=parent, **kwargs
        )
        self.node.transform = MatrixTransform()


class LayerOverlayMixin:
    layer: Layer

    def __init__(
        self,
        *,
        overlay,
        layer: Layer,
        node,
        viewer,
        parent=None,
        **kwargs,
    ) -> None:
        self.layer = layer
        super().__init__(
            node=node,
            overlay=overlay,
            viewer=viewer,
            parent=parent,
            **kwargs,
        )
        # need manual connection here because these overlays are not necessarily
        # always a child of the actual vispy node of the layer (eg, canvas overlays)
        self.layer.events.visible.connect(self._on_visible_change)

    def _should_be_visible(self) -> bool:
        return self.overlay.visible and self.layer.visible

    def close(self) -> None:
        disconnect_events(self.layer.events, self)
        super().close()


class ViewerOverlayMixin:
    viewer: ViewerModel

    def __init__(
        self,
        *,
        overlay,
        viewer: ViewerModel,
        node,
        parent=None,
        **kwargs,
    ) -> None:
        self.viewer = viewer
        super().__init__(
            node=node,
            overlay=overlay,
            viewer=viewer,
            parent=parent,
            **kwargs,
        )

    def close(self) -> None:
        disconnect_events(self.viewer.events, self)
        super().close()
