from __future__ import annotations

from typing import Any

from napari._vispy.overlays.base import (
    LayerOverlayMixin,
    ViewerOverlayMixin,
    VispyCanvasOverlay,
)
from napari._vispy.visuals.text import Text
from napari.components._viewer_constants import CanvasPosition
from napari.components.overlays import TextOverlay
from napari.settings import get_settings
from napari.utils.events import Event


class _VispyBaseTextOverlay(VispyCanvasOverlay):
    """Base class for vispy text overlays."""

    overlay: TextOverlay

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        self.node.font_size = self.overlay.font_size
        self.node.anchors = ('left', 'bottom')

        self.overlay.events.color.connect(self._on_color_change)
        self.overlay.events.box.connect(self._on_color_change)
        self.overlay.events.box_color.connect(self._on_color_change)
        self.overlay.events.font_size.connect(self._on_font_size_change)

        get_settings().appearance.events.theme.connect(self._on_color_change)
        self.viewer.events.theme.connect(self._on_color_change)

    def _connect_events(self) -> None:
        pass

    def _on_text_change(self) -> None:
        self.node.text = self.overlay.text
        self._on_position_change()

    def _on_visible_change(self) -> None:
        # ensure that dpi is updated when the scale bar is visible
        # this does not need to run _on_position_change because visibility
        # is already connected to the canvas callback by the canvas itself
        self._on_text_change()
        return super()._on_visible_change()

    def _on_color_change(self) -> None:
        self.node.color = (
            self.overlay.color
            if self.overlay.color is not None
            else self._get_fgcolor()
        )

    def _on_font_size_change(self) -> None:
        self.node.font_size = self.overlay.font_size
        self._on_position_change()

    def _on_position_change(self, event: Event | None = None) -> None:
        position = self.overlay.position
        anchors = ('left', 'bottom')
        if position == CanvasPosition.TOP_LEFT:
            anchors = ('left', 'bottom')
        elif position == CanvasPosition.TOP_RIGHT:
            anchors = ('right', 'bottom')
        elif position == CanvasPosition.TOP_CENTER:
            anchors = ('center', 'bottom')
        elif position == CanvasPosition.BOTTOM_RIGHT:
            anchors = ('right', 'top')
        elif position == CanvasPosition.BOTTOM_LEFT:
            anchors = ('left', 'top')
        elif position == CanvasPosition.BOTTOM_CENTER:
            anchors = ('center', 'top')

        self.node.anchors = anchors
        self.node.font_size = self.overlay.font_size

        self.x_size, self.y_size = self.node.get_width_height()

        # depending on the canvas position, we need to change the position of the anchor itself
        # to ensure the text aligns properly e.g. left when on the left, and right when on the right
        x = y = 0.0
        if anchors[0] == 'right':
            x = self.x_size
        elif anchors[0] == 'center':
            x = self.x_size / 2

        if anchors[1] == 'top':
            y = self.y_size

        self.node.pos = (x, y)

        super()._on_position_change()

    def reset(self) -> None:
        super().reset()
        self._on_text_change()
        self._on_color_change()
        self._on_font_size_change()


class _VispyViewerTextOverlay(ViewerOverlayMixin, _VispyBaseTextOverlay):
    def __init__(self, **kwargs: Any) -> None:
        font_manager = kwargs.get('font_manager')
        font_family = kwargs.get('font_family', 'OpenSans')
        super().__init__(
            node=Text(pos=(0, 0), font_manager=font_manager, face=font_family),
            **kwargs,
        )

        self._connect_events()
        self.reset()


class _VispyLayerTextOverlay(LayerOverlayMixin, _VispyBaseTextOverlay):
    def __init__(self, **kwargs: Any) -> None:
        font_manager = kwargs.get('font_manager')
        font_family = kwargs.get('font_family', 'OpenSans')
        super().__init__(
            node=Text(pos=(0, 0), font_manager=font_manager, face=font_family),
            **kwargs,
        )

        self._connect_events()
        self.reset()


class VispyTextOverlay(_VispyViewerTextOverlay):
    def _connect_events(self) -> None:
        self.overlay.events.text.connect(self._on_text_change)

    def _on_text_change(self) -> None:
        self.node.text = self.overlay.text
        self._on_position_change()


class VispyLayerNameOverlay(_VispyLayerTextOverlay):
    def _connect_events(self) -> None:
        self.layer.events.name.connect(self._on_text_change)

    def _on_text_change(self) -> None:
        self.node.text = self.layer.name
        self._on_position_change()


class VispyCurrentSliceOverlay(_VispyViewerTextOverlay):
    def _connect_events(self) -> None:
        self.viewer.dims.events.connect(self._on_text_change)

    def _on_text_change(self) -> None:
        dims = self.viewer.dims
        lines = []
        for dim in dims.not_displayed:
            position = dims.point[dim]
            label = dims.axis_labels[dim]
            lines.append(f'{label} = {position:.5g}')

        self.node.text = '\n'.join(lines)
        self._on_position_change()
