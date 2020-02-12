import numpy as np
from vispy.scene.visuals import Line, Compound
from .markers import Markers
from vispy.visuals.transforms import ChainTransform

from .vispy_base_layer import VispyBaseLayer
from ..utils.colormaps.standardize_color import transform_color


class VispyPointsLayer(VispyBaseLayer):
    _highlight_color = (0, 0.6, 1)
    _highlight_width = 5

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
            self.node = Compound([Markers(), Markers(), Line()])
        else:
            self.node = Compound([Markers(), Markers()])
        self.node.parent = parent
        self._reset_base()

    def _on_data_change(self, event=None):
        # Check if ndisplay has changed current node type needs updating
        if (
            self.layer.dims.ndisplay == 3 and len(self.node._subvisuals) != 2
        ) or (
            self.layer.dims.ndisplay == 2 and len(self.node._subvisuals) != 3
        ):
            self._on_display_change()
            self._on_highlight_change()

        if len(self.layer._data_view) > 0:
            edge_color = self.layer.edge_color[self.layer._indices_view]
            face_color = self.layer.face_color[self.layer._indices_view]
        else:
            edge_color = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
            face_color = np.array([[1.0, 1.0, 1.0, 1.0]], dtype=np.float32)

        # Set vispy data, noting that the order of the points needs to be
        # reversed to make the most recently added point appear on top
        # and the rows / columns need to be switch for vispys x / y ordering
        if len(self.layer._data_view) == 0:
            data = np.zeros((1, self.layer.dims.ndisplay))
            size = [0]
        else:
            data = self.layer._data_view
            size = self.layer._size_view

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
        self.node.update()

    def _on_highlight_change(self, event=None):
        if len(self.layer._highlight_index) > 0:
            # Color the hovered or selected points
            data = self.layer._data_view[self.layer._highlight_index]
            if data.ndim == 1:
                data = np.expand_dims(data, axis=0)
            size = self.layer._size_view[self.layer._highlight_index]
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
