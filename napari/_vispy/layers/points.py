import logging

import numpy as np

from napari.layers.points.points import _PointsSliceResponse

from ...settings import get_settings
from ...utils.colormaps.standardize_color import transform_color
from ...utils.events import disconnect_events
from ..utils.gl import BLENDING_MODES
from ..utils.text import update_text
from ..visuals.points import PointsVisual
from .base import VispyBaseLayer, _prepare_transform

LOGGER = logging.getLogger("napari._vispy.layers.points")


class VispyPointsLayer(VispyBaseLayer):
    _highlight_color = (0, 0.6, 1)
    _highlight_width = None

    def __init__(self, layer):
        self._highlight_width = get_settings().appearance.highlight_thickness

        node = PointsVisual()
        super().__init__(layer, node)

        self.layer.events.symbol.connect(self._on_symbol_change)
        self.layer.events.edge_width.connect(self._on_data_change)
        self.layer.events.edge_width_is_relative.connect(self._on_data_change)
        self.layer.events.edge_color.connect(self._on_data_change)
        self.layer._edge.events.colors.connect(self._on_data_change)
        self.layer._edge.events.color_properties.connect(self._on_data_change)
        self.layer.events.face_color.connect(self._on_data_change)
        self.layer._face.events.colors.connect(self._on_data_change)
        self.layer._face.events.color_properties.connect(self._on_data_change)
        self.layer.events.highlight.connect(self._on_highlight_change)
        self.layer.text.events.connect(self._on_text_change)
        self.layer.events.shading.connect(self._on_shading_change)
        self.layer.events.antialiasing.connect(self._on_antialiasing_change)
        self.layer.events.experimental_canvas_size_limits.connect(
            self._on_canvas_size_limits_change
        )

        self._on_data_change()

    # We upgrade the parameter type of this overridden method, which is
    # problematic for anything with a reference typed with the base Layer.
    # This is a code smell that should make us reconsider this design.
    def _set_slice(self, response: _PointsSliceResponse) -> None:
        LOGGER.debug('VispyPointsLayer._set_slice : %s', response.request)
        data = response.data[:, ::-1]

        if len(data) > 0:
            edge_color = response.edge_color
            face_color = response.face_color
            size = response.size
            edge_width = response.edge_width
        else:
            data = np.zeros((1, self.layer._ndisplay))
            edge_color = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
            face_color = np.array([[1.0, 1.0, 1.0, 1.0]], dtype=np.float32)
            size = [0]
            edge_width = [0]

        if response.edge_width_is_relative:
            edge_kw = {
                'edge_width': None,
                'edge_width_rel': edge_width,
            }
        else:
            edge_kw = {
                'edge_width': edge_width,
                'edge_width_rel': None,
            }

        self.node._subvisuals[0].set_data(
            response.data[:, ::-1],
            size=size,
            face_color=face_color,
            edge_color=edge_color,
            **edge_kw,
        )
        self._master_transform.matrix = _prepare_transform(
            response.data_to_world
        )

    def _on_data_change(self):
        if len(self.layer._indices_view) > 0:
            edge_color = self.layer._view_edge_color
            face_color = self.layer._view_face_color
        else:
            edge_color = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
            face_color = np.array([[1.0, 1.0, 1.0, 1.0]], dtype=np.float32)

        # Set vispy data, noting that the order of the points needs to be
        # reversed to make the most recently added point appear on top
        # and the rows / columns need to be switched for vispy's x / y ordering
        if len(self.layer._indices_view) == 0:
            data = np.zeros((1, self.layer._ndisplay))
            size = [0]
            edge_width = [0]
        else:
            data = self.layer._view_data
            size = self.layer._view_size
            edge_width = self.layer._view_edge_width

        set_data = self.node._subvisuals[0].set_data

        if self.layer.edge_width_is_relative:
            edge_kw = {
                'edge_width': None,
                'edge_width_rel': edge_width,
            }
        else:
            edge_kw = {
                'edge_width': edge_width,
                'edge_width_rel': None,
            }

        set_data(
            data[:, ::-1],
            size=size,
            **edge_kw,
            edge_color=edge_color,
            face_color=face_color,
        )

        self.reset()

    def _on_symbol_change(self):
        self.node.symbol = self.layer.symbol

    def _on_highlight_change(self):
        settings = get_settings()
        if len(self.layer._highlight_index) > 0:
            # Color the hovered or selected points
            data = self.layer._view_data[self.layer._highlight_index]
            if data.ndim == 1:
                data = np.expand_dims(data, axis=0)
            size = self.layer._view_size[self.layer._highlight_index]
        else:
            data = np.zeros((1, self.layer._ndisplay))
            size = 0

        self.node._subvisuals[1].set_data(
            data[:, ::-1],
            size=size,
            edge_width=settings.appearance.highlight_thickness,
            edge_color=self._highlight_color,
            face_color=transform_color('transparent'),
        )

        if (
            self.layer._highlight_box is None
            or 0 in self.layer._highlight_box.shape
        ):
            pos = np.zeros((1, self.layer._ndisplay))
            width = 0
        else:
            pos = self.layer._highlight_box
            width = settings.appearance.highlight_thickness

        self.node._subvisuals[2].set_data(
            pos=pos[:, ::-1],
            color=self._highlight_color,
            width=width,
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
        text_node = self.node._subvisuals[-1]
        return text_node

    def _on_text_change(self, event=None):
        if event is not None:
            if event.type == 'blending':
                self._on_blending_change(event)
                return
            if event.type == 'values':
                return
        self._update_text()

    def _on_blending_change(self):
        """Function to set the blending mode"""
        points_blending_kwargs = BLENDING_MODES[self.layer.blending]
        self.node.set_gl_state(**points_blending_kwargs)

        text_node = self._get_text_node()
        text_blending_kwargs = BLENDING_MODES[self.layer.text.blending]
        text_node.set_gl_state(**text_blending_kwargs)

        # selection box is always without depth
        box_blending_kwargs = BLENDING_MODES['translucent_no_depth']
        self.node._subvisuals[2].set_gl_state(**box_blending_kwargs)

        self.node.update()

    def _on_antialiasing_change(self):
        self.node.antialias = self.layer.antialiasing

    def _on_shading_change(self):
        shading = self.layer.shading
        if shading == 'spherical':
            self.node.spherical = True
        else:
            self.node.spherical = False

    def _on_canvas_size_limits_change(self):
        min_size, max_size = self.layer.experimental_canvas_size_limits
        self.node.clamp_filter.min_size = min_size
        self.node.clamp_filter.max_size = max_size

    def reset(self):
        super().reset()
        self._update_text(update_node=False)
        self._on_symbol_change()
        self._on_highlight_change()
        self._on_antialiasing_change()
        self._on_shading_change()
        self._on_canvas_size_limits_change()

    def close(self):
        """Vispy visual is closing."""
        disconnect_events(self.layer.text.events, self)
        super().close()
