from __future__ import annotations

import bisect
from decimal import Decimal
from math import floor, log
from typing import TYPE_CHECKING

import numpy as np
import pint

from napari._vispy.overlays.base import ViewerOverlayMixin, VispyCanvasOverlay
from napari._vispy.visuals.scale_bar import ScaleBar
from napari.utils._units import PREFERRED_VALUES
from napari.utils.notifications import show_warning

if TYPE_CHECKING:
    from napari._vispy.utils.qt_font import FontInfo
    from napari.components.overlays import ScaleBarOverlay


class VispyScaleBarOverlay(ViewerOverlayMixin, VispyCanvasOverlay):
    """Scale bar in world coordinates."""

    overlay: ScaleBarOverlay
    node: ScaleBar

    def __init__(self, *, font_info: FontInfo, **kwargs) -> None:
        self._canvas_length = 150.0
        self._canvas_tick_length = 10.0
        self._canvas_thickness = 3
        self._scale = 1.0
        self._unit = pint.Quantity('1 pixel')

        super().__init__(
            node=ScaleBar(font_info=font_info),
            font_info=font_info,
            **kwargs,
        )

        self.overlay.events.color.connect(self._on_rendering_change)
        self.overlay.events.colored.connect(self._on_rendering_change)
        self.overlay.events.box.connect(self._on_rendering_change)
        self.overlay.events.box_color.connect(self._on_rendering_change)
        self.overlay.events.font_size.connect(self._on_font_size_change)
        self.overlay.events.ticks.connect(self._on_rendering_change)
        self.overlay.events.unit.connect(self._on_unit_change)
        self.overlay.events.length.connect(self._on_size_or_zoom_change)
        self.overlay.events.visible.connect(self._on_rendering_change)
        self.overlay.events.gridded.connect(self._on_size_or_zoom_change)

        self.viewer.scene.camera.events.zoom.connect(
            self._on_size_or_zoom_change
        )
        self.viewer.dims.events.order.connect(self._on_unit_change)
        self.viewer.dims.events.ndisplay.connect(self._on_unit_change)
        self.viewer.canvas.events.background_color.connect(
            self._on_rendering_change
        )
        self.viewer.canvas.events.size.connect(self._on_size_or_zoom_change)
        self.viewer.canvas.grid.events.connect(self._on_size_or_zoom_change)

        self.reset()

    def _on_unit_change(self):
        # NOTE: this is also called by VispyCanvas when layer units are updated
        #       so it doesn't need to be connected to events for that
        if self.viewer.layers.units is not None:
            units = np.array(self.viewer.layers.units)[
                list(self.viewer.dims.displayed)
            ]
            if any(
                u.dimensionality != units[0].dimensionality for u in units[1:]
            ):
                dim_repr = tuple(str(d.dimensionality) for d in units)
                show_warning(
                    f'Displayed dimensions have mismatched dimensionality {dim_repr}. '
                    'The scale bar will only use the unit from the last displayed axis.',
                )
            unit = units[-1]
        else:
            unit = pint.get_application_registry()('dimensionless')
        self._unit = unit * 1  # convert unit to quantity
        self._on_size_or_zoom_change(force=True)

    def _on_font_size_change(self):
        self._on_size_or_zoom_change(force=True)

    def _calculate_best_length(
        self, desired_length: float
    ) -> tuple[float, pint.Quantity]:
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
        current_quantity = self._unit * desired_length
        # convert the value to compact representation
        new_quantity = current_quantity.to_compact()
        # calculate the scaling factor taking into account any conversion
        # that might have occurred (e.g. um -> cm)
        factor = current_quantity / new_quantity

        # select value closest to one of our preferred values and also
        # validate if quantity is dimensionless and lower than 1 to prevent
        # the scale bar to extend beyond the canvas when zooming.
        # If the value falls in those conditions, we use the corresponding
        # preferred value but scaled to take into account the actual value
        # magnitude. See https://github.com/napari/napari/issues/5914
        magnitude_1000 = floor(log(new_quantity.magnitude, 1000))
        scaled_magnitude = new_quantity.magnitude * 1000 ** (-magnitude_1000)
        index = bisect.bisect_left(PREFERRED_VALUES, scaled_magnitude)
        if index > 0:
            # When we get the lowest index of the list, removing -1 will
            # return the last index.
            index -= 1
        new_value: float = PREFERRED_VALUES[index]
        if new_quantity.dimensionless:
            # using Decimal is necessary to avoid `4.999999e-6`
            # at really small scale.
            new_value = float(
                Decimal(new_value) * Decimal(1000) ** magnitude_1000
            )

        # get the new pixel length utilizing the user-specified units
        new_length = (
            (new_value * factor) / (1 * self._unit).magnitude
        ).magnitude
        new_quantity = new_value * new_quantity.units
        return new_length, new_quantity

    def _on_size_or_zoom_change(self, *, force: bool = False):
        """Update length based on scale bar size and zoom."""

        scale = 1 / self.viewer.scene.camera.zoom

        if self.overlay.gridded:
            view_height, view_width = self.viewer.canvas.viewbox_size(
                self.viewer.layers
            )
        else:
            view_height, view_width = self.viewer.canvas.size

        target_canvas_length = view_width / 4
        # If scale or canvas size has not changed, do not redraw
        if (
            abs(np.log10(self._scale) - np.log10(scale)) < 1e-4
            and target_canvas_length == self._canvas_length
            and not force
        ):
            return

        # convert desired length to world size
        target_world_pixels = scale * target_canvas_length

        # If length is set, use that value to calculate the scale bar length
        if self.overlay.length is not None:
            canvas_length = self.overlay.length / scale
            dim_with_unit = self.overlay.length * self._unit.units
        else:
            # calculate the desired length as well as update the value and units
            target_world_pixels_rounded, dim_with_unit = (
                self._calculate_best_length(target_world_pixels)
            )
            canvas_length = target_world_pixels_rounded / scale

        self._canvas_length = canvas_length
        self._scale = scale

        # some magic numbers
        self._canvas_tick_length = max(view_height // 70, 11)
        self._canvas_thickness = max((view_width + view_height) // 700, 3)
        # prefer odd numbers as they look nicer
        self._canvas_tick_length += (self._canvas_tick_length + 1) % 2
        self._canvas_thickness += (self._canvas_thickness + 1) % 2

        # Update scalebar and text
        self.node.text.text = f'{dim_with_unit:g~#P}'
        self._on_rendering_change()

    def _on_rendering_change(self):
        """Change color and other rendering features of scale bar and box."""
        if not self.overlay.visible:
            return

        if self.overlay.colored:
            color = self.overlay.color
        else:
            color = self._get_fgcolor()

        width, height = self.node.set_data(
            length=self._canvas_length,
            tick_length=self._canvas_tick_length if self.overlay.ticks else 0,
            thickness=self._canvas_thickness,
            font_size=self.overlay.font_size,
            color=color,
        )

        size_changed = width != self.x_size or height != self.y_size
        self.x_size = width
        self.y_size = height
        if size_changed:
            self._on_position_change()

    def _on_visible_change(self):
        # ensure that dpi is updated when the scale bar is visible
        self._on_size_or_zoom_change()
        return super()._on_visible_change()

    def reset(self):
        super().reset()
        self._on_unit_change()
