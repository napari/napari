import numpy as np

from napari._vispy.overlays.base import ViewerOverlayMixin, VispyCanvasOverlay
from napari._vispy.visuals.slice_text import SliceText
from napari.utils.colormaps.standardize_color import transform_color
from napari.utils.theme import get_theme


class VispySliceTextOverlay(ViewerOverlayMixin, VispyCanvasOverlay):
    """Slice bar in world coordinates."""

    def __init__(self, *, viewer, overlay, parent=None) -> None:
        super().__init__(
            node=SliceText(), viewer=viewer, overlay=overlay, parent=parent
        )
        self.x_size = 5
        # need to change from defaults because the anchor is in the center
        self.y_offset = 50
        self.y_size = 10

        self.overlay.events.box.connect(self._on_box_change)

        self.overlay.events.box_color.connect(self._on_color_change)
        self.overlay.events.color.connect(self._on_color_change)
        self.overlay.events.colored.connect(self._on_color_change)
        self.viewer.events.theme.connect(self._on_color_change)

        self.overlay.events.font_size.connect(self._on_font_change)

        self.viewer.dims.events.point.connect(self._on_data_change)
        self.viewer.dims.events.ndisplay.connect(self._on_data_change)
        self.viewer.dims.events.axis_labels.connect(self._on_data_change)

        self.reset()

    def _on_color_change(self):
        color = self.overlay.color
        box_color = self.overlay.box_color

        if not self.overlay.colored:
            if self.overlay.box:
                # The box is visible - set the scale bar color to the negative of the
                # box color.
                color = 1 - box_color
                # color[-1] = 1
            else:
                # set scale color negative of theme background.
                # the reason for using the `as_hex` here is to avoid
                # `UserWarning` which is emitted when RGB values are above 1
                if self.node.parent is not None and self.node.parent.bgcolor:
                    background_color = self.node.parent.bgcolor.rgba
                else:
                    background_color = get_theme(
                        self.viewer.theme
                    ).canvas.as_hex()
                    background_color = transform_color(background_color)[0]
                color = np.subtract(1, background_color)
                # color[-1] = background_color[-1]  # exactly as ScaleBar, but not working as intended

        color[-1] = 1
        self.node.box.color = box_color
        self.node.text.color = color

    def _on_data_change(self):
        """Change color and data of scale bar and box."""
        not_displayed_dims = self.viewer.dims.not_displayed

        if len(not_displayed_dims) > 0:
            text = 'Current slice:\n'
            for dim in not_displayed_dims:
                num = self.viewer.dims.point[dim]
                name = self.viewer.dims.axis_labels[dim]
                formatted_num = format(num, '.5f').rstrip('0').rstrip('.')
                text += f'{name}={formatted_num}\n'
        else:
            text = ''

        self.node.text.text = text

    def _on_box_change(self):
        self.node.box.visible = self.overlay.box

    def _on_font_change(self):
        """Update font information"""
        self.node.text.font_size = self.overlay.font_size

    def reset(self):
        super().reset()
        self._on_color_change()
        self._on_data_change()
        self._on_box_change()
        self._on_font_change()
