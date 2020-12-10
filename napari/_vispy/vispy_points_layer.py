import numpy as np
from vispy.scene.visuals import Compound, Line, Text

from ..utils.colormaps.standardize_color import transform_color
from ..utils.events import disconnect_events
from ._text_utils import update_text
from .markers import Markers
from .vispy_base_layer import VispyBaseLayer


class VispyPointsLayer(VispyBaseLayer):
    _highlight_color = (0, 0.6, 1)
    _highlight_width = 2

    def __init__(self, layer):
        # Create a compound visual with the following four subvisuals:
        # Lines: The lines of the interaction box used for highlights.
        # Markers: The the outlines for each point used for highlights.
        # Markers: The actual markers of each point.
        node = Compound([Markers(), Markers(), Line(), Text()])

        super().__init__(layer, node)

        self.layer.events.symbol.connect(self._on_data_change)
        self.layer.events.edge_width.connect(self._on_data_change)
        self.layer.events.edge_color.connect(self._on_data_change)
        self.layer.events.face_color.connect(self._on_data_change)
        self.layer.text._connect_update_events(
            self._on_text_change, self._on_blending_change
        )
        self.layer.events.highlight.connect(self._on_highlight_change)
        self._on_data_change()
        self._reset_base()

    def _on_data_change(self, event=None):
        if len(self.layer._indices_view) > 0:
            edge_color = self.layer._view_edge_color
            face_color = self.layer._view_face_color
        else:
            edge_color = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
            face_color = np.array([[1.0, 1.0, 1.0, 1.0]], dtype=np.float32)

        # Set vispy data, noting that the order of the points needs to be
        # reversed to make the most recently added point appear on top
        # and the rows / columns need to be switch for vispys x / y ordering
        if len(self.layer._indices_view) == 0:
            data = np.zeros((1, self.layer._ndisplay))
            size = [0]
        else:
            data = self.layer._view_data
            size = self.layer._view_size

        set_data = self.node._subvisuals[0].set_data

        set_data(
            data[:, ::-1],
            size=size,
            edge_width=self.layer.edge_width,
            symbol=self.layer.symbol,
            edge_color=edge_color,
            face_color=face_color,
            scaling=True,
        )

        self._on_text_change()
        self.node.update()

        # Call to update order of translation values with new dims:
        self._on_matrix_change()

    def _on_highlight_change(self, event=None):
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
            edge_width=self._highlight_width,
            symbol=self.layer.symbol,
            edge_color=self._highlight_color,
            face_color=transform_color('transparent'),
            scaling=True,
        )

        # only draw a box in 2D
        if self.layer._ndisplay == 2:
            if (
                self.layer._highlight_box is None
                or 0 in self.layer._highlight_box.shape
            ):
                pos = np.zeros((1, self.layer._ndisplay))
                width = 0
            else:
                pos = self.layer._highlight_box
                width = self._highlight_width

            self.node._subvisuals[2].set_data(
                pos=pos[:, ::-1], color=self._highlight_color, width=width,
            )
        else:
            self.node._subvisuals[2].set_data(
                pos=np.zeros((1, self.layer._ndisplay)), width=0,
            )

        self.node.update()

    def _on_text_change(self, update_node=True):
        """Function to update the text node properties

        Parameters
        ----------
        update_node : bool
            If true, update the node after setting the properties
        """
        ndisplay = self.layer._ndisplay
        if (len(self.layer._indices_view) == 0) or (
            self.layer._text.visible is False
        ):
            text_coords = np.zeros((1, ndisplay))
            text = []
            anchor_x = 'center'
            anchor_y = 'center'
        else:
            text_coords, anchor_x, anchor_y = self.layer._view_text_coords
            if len(text_coords) == 0:
                text_coords = np.zeros((1, ndisplay))
            text = self.layer._view_text
        text_node = self._get_text_node()
        update_text(
            text_values=text,
            coords=text_coords,
            anchor=(anchor_x, anchor_y),
            rotation=self.layer._text.rotation,
            color=self.layer._text.color,
            size=self.layer._text.size,
            ndisplay=ndisplay,
            text_node=text_node,
        )

        if update_node:
            self.node.update()

    def _get_text_node(self):
        """Function to get the text node from the Compound visual"""
        text_node = self.node._subvisuals[-1]
        return text_node

    def _on_blending_change(self, event=None):
        """Function to set the blending mode"""
        self.node.set_gl_state(self.layer.blending)

        text_node = self._get_text_node()
        text_node.set_gl_state(self.layer.text.blending)
        self.node.update()

    def close(self):
        """Vispy visual is closing."""
        disconnect_events(self.layer.text.events, self)
        super().close()
