from vispy.scene.visuals import Text

from napari._vispy.overlays.base import (
    LayerOverlayMixin,
    ViewerOverlayMixin,
    VispyCanvasOverlay,
)
from napari._vispy.utils.text import get_text_width_height
from napari.components._viewer_constants import CanvasPosition


class _VispyBaseTextOverlay(VispyCanvasOverlay):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.node.font_size = self.overlay.font_size
        self.node.anchors = ('left', 'top')

        self.overlay.events.color.connect(self._on_color_change)
        self.overlay.events.font_size.connect(self._on_font_size_change)

    def _on_text_change(self):
        pass

    def _on_color_change(self):
        self.node.color = self.overlay.color

    def _on_font_size_change(self):
        self.node.font_size = self.overlay.font_size

    def _on_position_change(self, event=None):
        position = self.overlay.position

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

        self.x_size, self.y_size = get_text_width_height(self.node)

        x = y = 0.0
        if anchors[0] == 'right':
            x = self.x_size
        elif anchors[0] == 'center':
            x = self.x_size / 2

        if anchors[1] == 'top':
            y = self.y_size

        self.node.pos = (x, y)

        super()._on_position_change()

    def reset(self):
        super().reset()
        self._on_text_change()
        self._on_color_change()
        self._on_font_size_change()


class _VispyViewerTextOverlay(ViewerOverlayMixin, _VispyBaseTextOverlay):
    def __init__(self, **kwargs):
        super().__init__(node=Text(pos=(0, 0)), **kwargs)

        self._connect_events()
        self.reset()


class _VispyLayerTextOverlay(LayerOverlayMixin, _VispyBaseTextOverlay):
    def __init__(self, **kwargs):
        super().__init__(node=Text(pos=(0, 0)), **kwargs)

        self._connect_events()
        self.reset()


class VispyTextOverlay(_VispyViewerTextOverlay):
    def _connect_events(self):
        self.overlay.events.text.connect(self._on_text_change)

    def _on_text_change(self):
        self.node.text = self.overlay.text


class VispyLayerNameOverlay(_VispyLayerTextOverlay):
    def _connect_events(self):
        self.layer.events.name.connect(self._on_text_change)

    def _on_text_change(self):
        self.node.text = self.layer.name
        self._on_position_change()


class VispyDimsOverlay(_VispyViewerTextOverlay):
    def _connect_events(self):
        self.viewer.dims.events.connect(self._on_text_change)

    def _on_text_change(self):
        dims = self.viewer.dims
        lines = []
        for dim in dims.not_displayed:
            position = dims.point[dim]
            label = dims.axis_labels[dim]
            lines.append(f'{label} = {position:.5g}')

        self.node.text = '\n'.join(lines)
        self._on_position_change()
