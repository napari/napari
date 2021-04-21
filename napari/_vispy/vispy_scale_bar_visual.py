import bisect

import numpy as np
from vispy.scene.visuals import Line, Text
from vispy.visuals.transforms import STTransform

from ..components._viewer_constants import Position
from ..utils.colormaps.standardize_color import transform_color
from ..utils.theme import get_theme
from ..utils.translations import trans


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
        self._target_length = 150
        self._scale = 1
        self._px_size = 1
        _Dimension, current_unit = DIMENSIONS["pixel-length"]
        self._dimension: Dimension = _Dimension.from_unit(current_unit)

        self.node = Line(
            connect='segments', method='gl', parent=parent, width=3
        )
        self.node.order = order
        self.node.transform = STTransform()

        self.text_node = Text(pos=[0.5, -1], parent=self.node)
        self.text_node.order = order
        self.text_node.transform = STTransform()
        self.text_node.font_size = 10
        self.text_node.anchors = ("center", "center")
        self.text_node.text = f"{1}px"

        self._viewer.events.theme.connect(self._on_data_change)
        self._viewer.scale_bar.events.visible.connect(self._on_visible_change)
        self._viewer.scale_bar.events.colored.connect(self._on_data_change)
        self._viewer.scale_bar.events.ticks.connect(self._on_data_change)
        self._viewer.scale_bar.events.position.connect(
            self._on_position_change
        )
        self._viewer.camera.events.zoom.connect(self._on_zoom_change)
        self._viewer.scale_bar.events.font_size.connect(self._on_text_change)
        self._viewer.scale_bar.events.px_size.connect(self._on_dimension_change)
        self._viewer.scale_bar.events.dimension.connect(self._on_dimension_change)

        self._on_visible_change(None)
        self._on_data_change(None)
        self._on_dimension_change(None)
        self._on_position_change(None)
        self._on_dimension_change(None)

    def _on_dimension_change(self, _evt=None):
        """Update dimension"""
        self._px_size = 1 if self._viewer.scale_bar.dimension == "pixel-length" else self._viewer.scale_bar.px_size
        _Dimension, current_unit = DIMENSIONS[self._viewer.scale_bar.dimension]
        self._dimension = _Dimension.from_unit(current_unit)
        self._on_zoom_change(None, True)

    def _calculate_best_length(self, px_length):
        px_size = self._px_size
        value = px_length * px_size

        new_value, new_units = self._dimension.calculate_preferred(value)
        factor = value / new_value

        index = bisect.bisect_left(PREFERRED_VALUES, new_value)
        if index > 0:
            # When we get the lowest index of the list, removing -1 will
            # return the last index.
            index -= 1
        new_value = PREFERRED_VALUES[index]

        length_px = new_value * factor / px_size
        return length_px, new_value, new_units

    def _on_zoom_change(self, _evt=None, force: bool = False):
        """Update axes length based on zoom scale."""
        if not self._viewer.scale_bar.visible:
            return

        scale = 1 / self._viewer.camera.zoom

        # If scale has not changed, do not redraw
        if abs(np.log10(self._scale) - np.log10(scale)) < 1e-4 and not force:
            return
        self._scale = scale

        scale_canvas2world = self._scale
        target_canvas_pixels = self._target_length
        target_world_pixels = scale_canvas2world * target_canvas_pixels

        target_world_pixels_rounded, new_value, new_units = self._calculate_best_length(target_world_pixels)
        target_canvas_pixels_rounded = target_world_pixels_rounded / scale_canvas2world
        scale = target_canvas_pixels_rounded

        sign = -1 if self._viewer.scale_bar.position in [Position.TOP_RIGHT, Position.BOTTOM_RIGHT] else 1

        # Update scalebar and text
        self.node.transform.scale = [sign * scale, 1, 1, 1]
        self.text_node.text = f"{new_value}{new_units}"

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
        """Change visibility of scale bar."""
        self.node.visible = self._viewer.scale_bar.visible
        self.text_node.visible = self._viewer.scale_bar.visible
        self._on_zoom_change(None)

    def _on_text_change(self, event):
        """Update text information"""
        self.text_node.font_size = self._viewer.scale_bar.font_size

    def _on_position_change(self, _evt=None):
        """Change position of scale bar."""
        position = self._viewer.scale_bar.position
        x_bar_offset = 10
        canvas_size = list(self.node.canvas.size)

        if position == Position.TOP_LEFT:
            sign = 1
            bar_transform = [x_bar_offset, 10, 0, 0]
        elif position == Position.TOP_RIGHT:
            sign = -1
            bar_transform = [canvas_size[0] - x_bar_offset, 10, 0, 0]
        elif position == Position.BOTTOM_RIGHT:
            sign = -1
            bar_transform = [canvas_size[0] - x_bar_offset, canvas_size[1] - 30, 0, 0]
        elif position == Position.BOTTOM_LEFT:
            sign = 1
            bar_transform = [x_bar_offset, canvas_size[1] - 30, 0, 0]
        else:
            raise ValueError(
                trans._(
                    'Position {position} not recognized.',
                    deferred=True,
                    position=self._viewer.scale_bar.position,
                )
            )

        self.node.transform.translate = bar_transform
        scale = abs(self.node.transform.scale[0])
        self.node.transform.scale = [sign * scale, 1, 1, 1]
        self.text_node.transform.translate = (0, 20, 0, 0)
