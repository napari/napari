import numpy as np
from vispy.scene.visuals import Line, Compound, Text
from .markers import Markers
from vispy.visuals.transforms import ChainTransform

from .vispy_base_layer import VispyBaseLayer
from ..utils.colormaps.standardize_color import transform_color


class VispyPointsLayer(VispyBaseLayer):
    _highlight_color = (0, 0.6, 1)
    _highlight_width = 2

    def __init__(self, layer):
        # Create a compound visual with the following four subvisuals:
        # Lines: The lines of the interaction box used for highlights.
        # Markers: The the outlines for each point used for highlights.
        # Markers: The actual markers of each point.
        node = Compound([Markers(), Markers(), Line()])

        super().__init__(layer, node)

        self.layer.events.symbol.connect(self._on_data_change)
        self.layer.events.edge_width.connect(self._on_data_change)
        self.layer.events.edge_color.connect(self._on_data_change)
        self.layer.events.face_color.connect(self._on_data_change)
        self.layer.events.highlight.connect(self._on_highlight_change)
        self._on_display_change()
        self._on_data_change()

    def _on_display_change(self):
        parent = self.node.parent
        self.node.transforms = ChainTransform()
        self.node.parent = None

        if self.layer.dims.ndisplay == 2:
            self.node = Compound([Markers(), Markers(), Line(), Text()])
        else:
            self.node = Compound([Markers(), Markers(), Text()])
        self.node.parent = parent
        self._reset_base()

    def _on_data_change(self, event=None):
        # Check if ndisplay has changed current node type needs updating
        if (
            self.layer.dims.ndisplay == 3 and len(self.node._subvisuals) != 3
        ) or (
            self.layer.dims.ndisplay == 2 and len(self.node._subvisuals) != 4
        ):
            self._on_display_change()
            self._on_highlight_change()

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
            data = np.zeros((1, self.layer.dims.ndisplay))
            size = [0]
        else:
            data = self.layer._view_data
            size = self.layer._view_size

        set_data = self.node._subvisuals[0].set_data

        set_data(
            data[:, ::-1] + 0.5,
            size=size,
            edge_width=self.layer.edge_width,
            symbol=self.layer.symbol,
            edge_color=edge_color,
            face_color=face_color,
            scaling=True,
        )

        # update text
        if len(self.layer._indices_view) == 0:
            text_coords = np.zeros((1, self.layer.dims.ndisplay))
            text = []
        else:
            text_coords = self.layer._view_text_coords
            text = self.layer._view_text

        if self.layer.dims.ndisplay == 2:
            positions = np.flip(text_coords, axis=1)
        elif self.layer.dims.ndisplay == 3:
            raw_positions = np.flip(text_coords, axis=1)
            n_positions, position_dims = raw_positions.shape

            if position_dims < 3:
                padded_positions = np.zeros((n_positions, 3))
                padded_positions[:, 0:2] = raw_positions
                positions = padded_positions
            else:
                positions = raw_positions

        text_node = self.node._subvisuals[-1]
        self._update_text_node(
            text_node,
            text=text,
            pos=positions,
            rotation=self.layer._text.rotation,
            color=self.layer._text.color,
            font_size=self.layer._text.size,
        )
        self.node.update()
        # Call to update order of translation values with new dims:
        self._on_scale_change()
        self._on_translate_change()

    def _on_highlight_change(self, event=None):
        if len(self.layer._highlight_index) > 0:
            # Color the hovered or selected points
            data = self.layer._view_data[self.layer._highlight_index]
            if data.ndim == 1:
                data = np.expand_dims(data, axis=0)
            size = self.layer._view_size[self.layer._highlight_index]
        else:
            data = np.zeros((1, self.layer.dims.ndisplay))
            size = 0

        self.node._subvisuals[1].set_data(
            data[:, ::-1] + 0.5,
            size=size,
            edge_width=self._highlight_width,
            symbol=self.layer.symbol,
            edge_color=self._highlight_color,
            face_color=transform_color('transparent'),
            scaling=True,
        )

        # only draw a box in 2D
        if self.layer.dims.ndisplay == 2:
            if (
                self.layer._highlight_box is None
                or 0 in self.layer._highlight_box.shape
            ):
                pos = np.zeros((1, self.layer.dims.ndisplay))
                width = 0
            else:
                pos = self.layer._highlight_box
                width = self._highlight_width

            self.node._subvisuals[2].set_data(
                pos=pos[:, ::-1] + 0.5,
                color=self._highlight_color,
                width=width,
            )

        self.node.update()

    def _update_text_node(
        self, node, text=[], rotation=0, color='black', font_size=12, pos=None
    ):
        node.text = text
        node.pos = pos
        node.rotation = rotation
        node.color = color
        node.font_size = font_size

        node.update()
