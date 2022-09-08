"""Scale Bar overlay."""
import bisect

import numpy as np
from vispy.scene.visuals import Line, Rectangle, Text
from vispy.visuals.transforms import STTransform

from ...components._viewer_constants import Position
from ...utils._units import PREFERRED_VALUES, get_unit_registry
from ...utils.colormaps.standardize_color import transform_color
from ...utils.theme import get_theme
from ...utils.translations import trans


class VispyScaleBarOverlay:
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
        self._target_length = 150
        self._scale = 1
        self._quantity = None
        self._unit_reg = None

        self.line_node = Line(
            connect='segments', method='gl', parent=parent, width=3
        )
        self.line_node.order = order + 1
        self.line_node.transform = STTransform()

        # In order for the text and box to always appear centered on the scale
        # bar, the text and rect nodes should use the line node as the parent.
        self.text_node = Text(pos=[0.5, -1], parent=self.line_node)
        self.text_node.order = order + 1
        self.text_node.transform = STTransform()
        self.text_node.font_size = 10
        self.text_node.anchors = ("center", "center")
        self.text_node.text = "1px"

        self.rect_node = Rectangle(
            center=[0.5, 0.5],
            width=1.1,
            height=36,
            color=self._viewer.scale_bar.box_color,
            parent=self.line_node,
        )
        self.rect_node.order = order
        self.rect_node.transform = STTransform()

        # the two canvas are not the same object, better be safe.
        self.rect_node.canvas._backend.destroyed.connect(self._set_canvas_none)
        self.line_node.canvas._backend.destroyed.connect(self._set_canvas_none)
        self.text_node.canvas._backend.destroyed.connect(self._set_canvas_none)
        assert self.rect_node.canvas is self.line_node.canvas
        assert self.line_node.canvas is self.text_node.canvas
        # End Note

        self._viewer.events.theme.connect(self._on_data_change)
        self._viewer.scale_bar.events.visible.connect(self._on_visible_change)
        self._viewer.scale_bar.events.colored.connect(self._on_data_change)
        self._viewer.scale_bar.events.ticks.connect(self._on_data_change)
        self._viewer.scale_bar.events.box_color.connect(self._on_data_change)
        self._viewer.scale_bar.events.color.connect(self._on_data_change)
        self._viewer.scale_bar.events.position.connect(
            self._on_position_change
        )
        self._viewer.camera.events.zoom.connect(self._on_zoom_change)
        self._viewer.scale_bar.events.font_size.connect(self._on_text_change)
        self._viewer.scale_bar.events.unit.connect(self._on_dimension_change)
        self._viewer.scale_bar.events.box.connect(self._on_visible_change)

        self._on_visible_change()
        self._on_data_change()
        self._on_dimension_change()
        self._on_position_change()

    def _set_canvas_none(self):
        self.rect_node._set_canvas(None)
        self.line_node._set_canvas(None)
        self.text_node._set_canvas(None)

    @property
    def unit_registry(self):
        """Get unit registry.

        Rather than instantiating UnitRegistry earlier on, it is instantiated
        only when it is needed. The reason for this is that importing `pint`
        at module level can be time consuming.

        Notes
        -----
        https://github.com/napari/napari/pull/2617#issuecomment-827716325
        https://github.com/napari/napari/pull/2325
        """
        if self._unit_reg is None:
            self._unit_reg = get_unit_registry()
        return self._unit_reg

    def _on_dimension_change(self):
        """Update dimension."""
        if not self._viewer.scale_bar.visible and self._unit_reg is None:
            return
        unit = self._viewer.scale_bar.unit
        self._quantity = self.unit_registry(unit)
        self._on_zoom_change(force=True)

    def _calculate_best_length(self, desired_length: float):
        """Calculate new quantity based on the pixel length of the bar.

        Parameters
        ----------
        desired_length : float
            Desired length of the scale bar in world size.

        Returns
        -------
        new_length : float
            New length of the scale bar in world size based
            on the preferred scale bar value.
        new_quantity : pint.Quantity
            New quantity with abbreviated base unit.
        """
        current_quantity = self._quantity * desired_length
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
        new_length = (
            (new_value * factor) / self._quantity.magnitude
        ).magnitude
        new_quantity = new_value * new_quantity.units
        return new_length, new_quantity

    def _on_zoom_change(self, *, force: bool = False):
        """Update axes length based on zoom scale."""
        if not self._viewer.scale_bar.visible:
            return

        # If scale has not changed, do not redraw
        scale = 1 / self._viewer.camera.zoom
        if abs(np.log10(self._scale) - np.log10(scale)) < 1e-4 and not force:
            return
        self._scale = scale

        scale_canvas2world = self._scale
        target_canvas_pixels = self._target_length
        # convert desired length to world size
        target_world_pixels = scale_canvas2world * target_canvas_pixels

        # calculate the desired length as well as update the value and units
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
        self.line_node.transform.scale = [sign * scale, 1, 1, 1]
        self.text_node.text = f'{new_dim:~}'

    def _on_data_change(self):
        """Change color and data of scale bar and box."""
        color = self._viewer.scale_bar.color
        box_color = self._viewer.scale_bar.box_color

        if not self._viewer.scale_bar.colored:
            if self._viewer.scale_bar.box:
                # The box is visible - set the scale bar color to the negative of the
                # box color.
                color = 1 - box_color
                color[-1] = 1
            else:
                # set scale color negative of theme background.
                # the reason for using the `as_hex` here is to avoid
                # `UserWarning` which is emitted when RGB values are above 1
                background_color = get_theme(
                    self._viewer.theme, False
                ).canvas.as_hex()
                background_color = transform_color(background_color)[0]
                color = np.subtract(1, background_color)
                color[-1] = background_color[-1]

        if self._viewer.scale_bar.ticks:
            data = self._data
        else:
            data = self._data[:2]

        self.line_node.set_data(data, color)
        self.text_node.color = color
        self.rect_node.color = box_color

    def _on_visible_change(self):
        """Change visibility of scale bar."""
        self.rect_node.visible = (
            self._viewer.scale_bar.visible and self._viewer.scale_bar.box
        )
        self.line_node.visible = self._viewer.scale_bar.visible
        self.text_node.visible = self._viewer.scale_bar.visible

        # update unit if scale bar is visible and quantity
        # has not been specified yet or current unit is not
        # equivalent
        if self._viewer.scale_bar.visible and (
            self._quantity is None
            or self._quantity.units != self._viewer.scale_bar.unit
        ):
            self._quantity = self.unit_registry(self._viewer.scale_bar.unit)
        # only force zoom update if the scale bar is visible
        self._on_zoom_change(force=self._viewer.scale_bar.visible)

    def _on_text_change(self):
        """Update text information"""
        self.text_node.font_size = self._viewer.scale_bar.font_size

    def _on_position_change(self, _event=None):
        """Change position of scale bar."""
        position = self._viewer.scale_bar.position
        x_bar_offset, y_bar_offset = 10, 30
        canvas_size = list(self.rect_node.canvas.size)

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
                canvas_size[1] - y_bar_offset,
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

        self.line_node.transform.translate = bar_transform
        scale = abs(self.line_node.transform.scale[0])
        self.line_node.transform.scale = [sign * scale, 1, 1, 1]
        self.rect_node.transform.translate = (0, 10, 0, 0)
        self.text_node.transform.translate = (0, 20, 0, 0)
