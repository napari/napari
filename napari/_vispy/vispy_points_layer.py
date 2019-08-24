import numpy as np
from vispy.scene.visuals import Line, Markers, Compound

from ..layers import Points
from .vispy_base_layer import VispyBaseLayer


class VispyPointsLayer(VispyBaseLayer, layer=Points):
    _highlight_color = (0, 0.6, 1)
    _highlight_width = 1.5

    def __init__(self, layer):
        # Create a compound visual with the following four subvisuals:
        # Lines: The lines of the interaction box used for highlights.
        # Markers: The the outlines for each point used for highlights.
        # Markers: The actual markers of each point.
        node = Compound([Line(), Markers(), Markers()])

        super().__init__(layer, node)

        self.layer.events.symbol.connect(lambda e: self._on_data_change())
        self.layer.events.edge_width.connect(lambda e: self._on_data_change())
        self.layer.events.edge_color.connect(lambda e: self._on_data_change())
        self.layer.events.face_color.connect(lambda e: self._on_data_change())
        self.layer.events.highlight.connect(
            lambda e: self._on_highlight_change()
        )

        self.reset()

    def _on_data_change(self):
        if len(self.layer._data_view) > 0:
            edge_color = [
                self.layer.edge_colors[i]
                for i in self.layer._indices_view[::-1]
            ]
            face_color = [
                self.layer.face_colors[i]
                for i in self.layer._indices_view[::-1]
            ]
        else:
            edge_color = 'white'
            face_color = 'white'

        # Set vispy data, noting that the order of the points needs to be
        # reversed to make the most recently added point appear on top
        # and the rows / columns need to be switch for vispys x / y ordering
        self.node._subvisuals[2].set_data(
            self.layer._data_view[::-1, ::-1] + 0.5,
            size=self.layer._sizes_view[::-1],
            edge_width=self.layer.edge_width,
            symbol=self.layer.symbol,
            edge_color=edge_color,
            face_color=face_color,
            scaling=True,
        )
        self.node.update()

    def _on_highlight_change(self):
        if len(self.layer._highlight_index) > 0:
            # Color the hovered or selected points
            data = self.layer._data_view[self.layer._highlight_index]
            if data.ndim == 1:
                data = np.expand_dims(data, axis=0)
            size = self.layer._sizes_view[self.layer._highlight_index]
            face_color = [
                self.layer.face_colors[i]
                for i in self.layer._indices_view[self.layer._highlight_index]
            ]
        else:
            data = np.empty((0, 2))
            size = 1
            face_color = 'white'

        self.node._subvisuals[1].set_data(
            data[:, ::-1] + 0.5,
            size=size,
            edge_width=self._highlight_width,
            symbol=self.layer.symbol,
            edge_color=self._highlight_color,
            face_color=face_color,
            scaling=True,
        )

        self.node._subvisuals[0].set_data(
            pos=self.layer._highlight_box[:, [1, 0]] + 0.5,
            color=self._highlight_color,
            width=self._highlight_width,
        )

    def reset(self):
        self._reset_base()
        self._on_data_change()
        self._on_highlight_change()
