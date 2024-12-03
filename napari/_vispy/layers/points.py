import numpy as np

from napari._vispy.layers.base import VispyBaseLayer
from napari._vispy.utils.gl import BLENDING_MODES
from napari._vispy.utils.text import update_text
from napari._vispy.visuals.points import PointsVisual
from napari.settings import get_settings
from napari.utils.colormaps.standardize_color import transform_color
from napari.utils.events import disconnect_events


class VispyPointsLayer(VispyBaseLayer):
    node: PointsVisual

    def __init__(self, layer) -> None:
        node = PointsVisual()
        super().__init__(layer, node)

        self.layer.events.symbol.connect(self._on_data_change)
        self.layer.events.border_width.connect(self._on_data_change)
        self.layer.events.border_width_is_relative.connect(
            self._on_data_change
        )
        self.layer.events.border_color.connect(self._on_data_change)
        self.layer._border.events.colors.connect(self._on_data_change)
        self.layer._border.events.color_properties.connect(
            self._on_data_change
        )
        self.layer.events.face_color.connect(self._on_data_change)
        self.layer._face.events.colors.connect(self._on_data_change)
        self.layer._face.events.color_properties.connect(self._on_data_change)
        self.layer.events.highlight.connect(self._on_highlight_change)
        self.layer.text.events.connect(self._on_text_change)
        self.layer.events.shading.connect(self._on_shading_change)
        self.layer.events.antialiasing.connect(self._on_antialiasing_change)
        self.layer.events.canvas_size_limits.connect(
            self._on_canvas_size_limits_change
        )

        self._on_data_change()

    def _on_data_change(self):
        # Set vispy data, noting that the order of the points needs to be
        # reversed to make the most recently added point appear on top
        # and the rows / columns need to be switched for vispy's x / y ordering
        if len(self.layer._indices_view) == 0:
            # always pass one invisible point to avoid issues
            data = np.zeros((1, self.layer._slice_input.ndisplay))
            size = np.zeros(1)
            border_color = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
            face_color = np.array([[1.0, 1.0, 1.0, 1.0]], dtype=np.float32)
            border_width = np.zeros(1)
            symbol = ['o']
        else:
            data = self.layer._view_data
            size = self.layer._view_size
            border_color = self.layer._view_border_color
            face_color = self.layer._view_face_color
            border_width = self.layer._view_border_width
            symbol = [str(x) for x in self.layer._view_symbol]

        set_data = self.node.points_markers.set_data

        # use only last dimension to scale point sizes, see #5582
        scale = self.layer.scale[-1]

        if self.layer.border_width_is_relative:
            border_kw = {
                'edge_width': None,
                'edge_width_rel': border_width,
            }
        else:
            border_kw = {
                'edge_width': border_width * scale,
                'edge_width_rel': None,
            }

        set_data(
            data[:, ::-1],
            size=size * scale,
            symbol=symbol,
            # edge_color is the name of the vispy marker visual kwarg
            edge_color=border_color,
            face_color=face_color,
            **border_kw,
        )

        self.reset()

    def _on_highlight_change(self):
        settings = get_settings()
        if len(self.layer._highlight_index) > 0:
            # Color the hovered or selected points
            data = self.layer._view_data[self.layer._highlight_index]
            if data.ndim == 1:
                data = np.expand_dims(data, axis=0)
            size = self.layer._view_size[self.layer._highlight_index]
            border_width = self.layer._view_border_width[
                self.layer._highlight_index
            ]
            if self.layer.border_width_is_relative:
                border_width = (
                    border_width
                    * self.layer._view_size[self.layer._highlight_index][-1]
                )
            symbol = self.layer._view_symbol[self.layer._highlight_index]
        else:
            data = np.zeros((1, self.layer._slice_input.ndisplay))
            size = 0
            symbol = ['o']
            border_width = np.array([0])

        scale = self.layer.scale[-1]
        highlight_thickness = settings.appearance.highlight.highlight_thickness
        scaled_highlight = highlight_thickness * self.layer.scale_factor
        scaled_size = (size + border_width) * scale
        highlight_color = tuple(settings.appearance.highlight.highlight_color)

        self.node.selection_markers.set_data(
            data[:, ::-1],
            size=scaled_size,
            symbol=symbol,
            edge_width=scaled_highlight * 2,
            edge_color=highlight_color,
            face_color=transform_color('transparent'),
        )

        if (
            self.layer._highlight_box is None
            or 0 in self.layer._highlight_box.shape
        ):
            pos = np.zeros((1, self.layer._slice_input.ndisplay))
            highlight_thickness = 0
        else:
            pos = self.layer._highlight_box

        self.node.highlight_lines.set_data(
            pos=pos[:, ::-1],
            color=highlight_color,
            width=highlight_thickness,
        )

        self.node.update()

    def _update_text(self, *, update_node=True):
        """Function to update the text node properties

        Parameters
        ----------
        update_node : bool
            If true, update the node after setting the properties
        """
        update_text(node=self._get_text_node(), layer=self.layer)
        if update_node:
            self.node.update()

    def _get_text_node(self):
        """Function to get the text node from the Compound visual"""
        return self.node.text

    def _on_text_change(self, event=None):
        if event is not None:
            if event.type == 'blending':
                self._on_blending_change(event)
                return
            if event.type == 'values':
                return
        self._update_text()

    def _on_blending_change(self, event=None):
        """Function to set the blending mode"""
        points_blending_kwargs = BLENDING_MODES[self.layer.blending]
        self.node.set_gl_state(**points_blending_kwargs)

        text_node = self._get_text_node()
        text_blending_kwargs = BLENDING_MODES[self.layer.text.blending]
        text_node.set_gl_state(**text_blending_kwargs)

        # selection box is always without depth
        box_blending_kwargs = BLENDING_MODES['translucent_no_depth']
        self.node.highlight_lines.set_gl_state(**box_blending_kwargs)

        self.node.update()

    def _on_antialiasing_change(self):
        self.node.antialias = self.layer.antialiasing

    def _on_shading_change(self):
        shading = self.layer.shading
        self.node.spherical = shading == 'spherical'

    def _on_canvas_size_limits_change(self):
        self.node.points_markers.canvas_size_limits = (
            self.layer.canvas_size_limits
        )
        highlight_thickness = (
            get_settings().appearance.highlight.highlight_thickness
        )
        low, high = self.layer.canvas_size_limits
        self.node.selection_markers.canvas_size_limits = (
            low + highlight_thickness,
            high + highlight_thickness,
        )
        self.node.update()

    def reset(self):
        super().reset()
        self._update_text(update_node=False)
        self._on_highlight_change()
        self._on_antialiasing_change()
        self._on_shading_change()
        self._on_canvas_size_limits_change()

    def close(self):
        """Vispy visual is closing."""
        disconnect_events(self.layer.text.events, self)
        super().close()
