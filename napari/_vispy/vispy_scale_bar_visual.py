import numpy as np
from vispy.scene.visuals import Line, Text
from vispy.visuals.transforms import STTransform

from ..components._viewer_constants import Position
from ..utils.colormaps.standardize_color import transform_color
from ..utils.theme import get_theme


class VispyScaleBarVisual:
    """Scale bar in world coordinates."""

    def __init__(self, viewer, parent=None, order=0):
        self._viewer = viewer

        self._data = np.array(
            [
                [0, 0, -1],
                [1, 0, -1],
                [0, -5, -1],
                [0, 5, -1],
                [1, -5, -1],
                [1, 5, -1],
            ]
        )
        self._default_color = np.array([1, 0, 1, 1])
        self._target_length = 100
        self._scale = 1

        self.node = Line(
            connect='segments', method='gl', parent=parent, width=3
        )
        self.node.order = order
        self.node.transform = STTransform()

        self.text_node = Text(pos=[0, 0], parent=parent)
        self.text_node.order = order
        self.text_node.transform = STTransform()
        self.text_node.font_size = 10
        self.text_node.anchors = ('center', 'center')
        self.text_node.text = f'{1}'

        self._viewer.events.theme.connect(self._on_data_change)
        self._viewer.scale_bar.events.visible.connect(self._on_visible_change)
        self._viewer.scale_bar.events.colored.connect(self._on_data_change)
        self._viewer.scale_bar.events.ticks.connect(self._on_data_change)
        self._viewer.scale_bar.events.position.connect(
            self._on_position_change
        )
        self._viewer.camera.events.zoom.connect(self._on_zoom_change)

        self._on_visible_change(None)
        self._on_data_change(None)
        self._on_position_change(None)

    def _on_zoom_change(self, event):
        """Update axes length based on zoom scale."""
        if not self._viewer.scale_bar.visible:
            return

        scale = 1 / self._viewer.camera.zoom

        # If scale has not changed, do not redraw
        if abs(np.log10(self._scale) - np.log10(scale)) < 1e-4:
            return
        self._scale = scale

        scale_canvas2world = self._scale
        target_canvas_pixels = self._target_length
        target_world_pixels = scale_canvas2world * target_canvas_pixels

        # Round scalebar to nearest factor of 1, 2, 5, 10 in world pixels on a log scale
        resolutions = [1, 2, 5, 10]
        log_target = np.log10(target_world_pixels)
        mult_ind = np.argmin(
            np.abs(np.subtract(np.log10(resolutions), log_target % 1))
        )
        power_val = np.floor(log_target)
        target_world_pixels_rounded = resolutions[mult_ind] * np.power(
            10, power_val
        )

        # Convert back to canvas pixels to get actual length of scale bar
        target_canvas_pixels_rounded = (
            target_world_pixels_rounded / scale_canvas2world
        )
        scale = target_canvas_pixels_rounded

        if self._viewer.scale_bar.position in [
            Position.TOP_RIGHT,
            Position.BOTTOM_RIGHT,
        ]:
            sign = -1
        else:
            sign = 1

        # Update scalebar and text
        self.node.transform.scale = [sign * scale, 1, 1, 1]
        self.text_node.text = f'{target_world_pixels_rounded:.4g}'

    def _on_data_change(self, event):
        """Change color and data of scale bar."""
        if self._viewer.scale_bar.colored:
            color = self._default_color
        else:
            background_color = get_theme(self._viewer.theme)['canvas']
            background_color = transform_color(background_color)[0]
            color = np.subtract(1, background_color)
            color[-1] = background_color[-1]

        if self._viewer.scale_bar.ticks:
            data = self._data
        else:
            data = self._data[:2]

        self.node.set_data(data, color)
        self.text_node.color = color

    def _on_visible_change(self, event):
        """Change visibiliy of scale bar."""
        self.node.visible = self._viewer.scale_bar.visible
        self.text_node.visible = self._viewer.scale_bar.visible
        self._on_zoom_change(None)

    def _on_position_change(self, event):
        """Change position of scale bar."""
        if self._viewer.scale_bar.position == Position.TOP_LEFT:
            sign = 1
            self.node.transform.translate = [66, 14, 0, 0]
            self.text_node.transform.translate = [33, 16, 0, 0]
        elif self._viewer.scale_bar.position == Position.TOP_RIGHT:
            sign = -1
            canvas_size = list(self.node.canvas.size)
            self.node.transform.translate = [canvas_size[0] - 66, 14, 0, 0]
            self.text_node.transform.translate = [
                canvas_size[0] - 33,
                16,
                0,
                0,
            ]
        elif self._viewer.scale_bar.position == Position.BOTTOM_RIGHT:
            sign = -1
            canvas_size = list(self.node.canvas.size)
            self.node.transform.translate = [
                canvas_size[0] - 66,
                canvas_size[1] - 16,
                0,
                0,
            ]
            self.text_node.transform.translate = [
                canvas_size[0] - 33,
                canvas_size[1] - 14,
                0,
                0,
            ]
        elif self._viewer.scale_bar.position == Position.BOTTOM_LEFT:
            sign = 1
            canvas_size = list(self.node.canvas.size)
            self.node.transform.translate = [66, canvas_size[1] - 16, 0, 0]
            self.text_node.transform.translate = [
                33,
                canvas_size[1] - 14,
                0,
                0,
            ]
        else:
            raise ValueError(
                f'Position {self._viewer.scale_bar.position}'
                ' not recognized.'
            )

        scale = abs(self.node.transform.scale[0])
        self.node.transform.scale = [sign * scale, 1, 1, 1]
