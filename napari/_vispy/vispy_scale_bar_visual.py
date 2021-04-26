"""Scale Bar visual"""
import bisect

import numpy as np
from vispy.scene.visuals import Line, Text
from vispy.visuals.transforms import STTransform

from ..components._viewer_constants import Position
from ..utils._units import PREFERRED_VALUES, UNIT_REG
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
        self._quantity = UNIT_REG("")  # unit-less

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
        self._viewer.scale_bar.events.unit.connect(self._on_dimension_change)

        self._on_visible_change(None)
        self._on_data_change(None)
        self._on_dimension_change(None)
        self._on_position_change(None)
        self._on_dimension_change(None)

    def _on_dimension_change(self, _evt=None):
        """Update dimension"""
        self._quantity = UNIT_REG(self._viewer.scale_bar.unit)
        self._on_zoom_change(None, True)

    def _calculate_best_length(self, px_length: float):
        """calculate new quantity based on the pixel length of the bar."""
        current_quantity = self._quantity * px_length
        # convert the value to compact representation
        new_quantity = current_quantity.to_compact()
        # calculate the scaling factor taking into account any conversion
        # that might have occurred (e.g. um -> cm)
        factor = current_quantity / new_quantity

        # select value closest to one of our preferred values
        index = bisect.bisect_left(PREFERRED_VALUES, new_quantity.magnitude)
        if index > 0:
            # When we get the lowest index of the list, removing -1 will
            # return the last index.
            index -= 1
        new_value = PREFERRED_VALUES[index]

        # get the new pixel length utilizing the user-specified units
        length_px = ((new_value * factor) / self._quantity.magnitude).magnitude
        new_quantity = new_value * new_quantity.units
        return length_px, new_quantity

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
        # convert desired length to world size
        target_world_pixels = scale_canvas2world * target_canvas_pixels

        # calculate the desired length as well as update the value and new units
        target_world_pixels_rounded, new_dim = self._calculate_best_length(
            target_world_pixels
        )
        target_canvas_pixels_rounded = (
            target_world_pixels_rounded / scale_canvas2world
        )
        scale = target_canvas_pixels_rounded

        sign = (
            -1
            if self._viewer.scale_bar.position
            in [Position.TOP_RIGHT, Position.BOTTOM_RIGHT]
            else 1
        )

        # Update scalebar and text
        self.node.transform.scale = [sign * scale, 1, 1, 1]
        self.text_node.text = f'{new_dim:~}'

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
            bar_transform = [
                canvas_size[0] - x_bar_offset,
                canvas_size[1] - 30,
                0,
                0,
            ]
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
