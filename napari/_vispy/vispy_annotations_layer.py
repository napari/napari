import numpy as np
from copy import copy
from vispy.scene.visuals import Line, Compound, Text
from .markers import Markers
from vispy.visuals.transforms import ChainTransform

from ..layers import Annotations
from .vispy_base_layer import VispyBaseLayer
import numpy as np


class VispyAnnotationsLayer(VispyBaseLayer):
    _highlight_color = (0, 0.6, 1)
    _highlight_width = 1.5

    _OUTLINE_MARKERS_NODE_INDEX = 0
    _TEXT_NODE_INDEX = 1
    _HIGHLIGHT_TEXT_NODE_INDEX = 2

    def __init__(self, layer):
        # Create a compound visual with the following four subvisuals:
        # Lines: The lines of the interaction box used for highlights.
        # Markers: The the outlines for each point used for highlights.
        # Markers: The actual markers of each point.
        node = Compound([Line(), Text(), Text()])

        super().__init__(layer, node)

        self.layer.events.symbol.connect(lambda e: self._on_data_change())
        self.layer.events.edge_width.connect(lambda e: self._on_data_change())
        self.layer.events.edge_color.connect(lambda e: self._on_data_change())
        self.layer.events.face_color.connect(lambda e: self._on_data_change())
        self.layer.events.highlight.connect(
            lambda e: self._on_highlight_change()
        )

        self.layer.dims.events.ndisplay.connect(
            lambda e: self._on_display_change()
        )

        self._on_display_change()

    def _on_display_change(self):
        parent = self.node.parent
        self.node.transforms = ChainTransform()
        self.node.parent = None

        if self.layer.dims.ndisplay == 2:
            self.node = Compound([Markers(), Line(), Text(), Text()])
        else:
            self.node = Markers()

        self.node.parent = parent
        self.layer._update_dims()
        self.layer._set_view_slice()
        self.reset()

    def _on_data_change(self):
        if len(self.layer._data_view) > 0:
            edge_color = [
                self.layer.edge_colors[i] for i in self.layer._indices_view
            ]
            face_color = [
                self.layer.face_colors[i] for i in self.layer._indices_view
            ]
        else:
            edge_color = 'white'
            face_color = 'white'

        # Set vispy data, noting that the order of the points needs to be
        # reversed to make the most recently added point appear on top
        # and the rows / columns need to be switch for vispys x / y ordering
        if len(self.layer._data_view) == 0:
            data = np.zeros((1, self.layer.dims.ndisplay))
            size = [0]
        else:
            data = self.layer._data_view
            annotations = self.layer._annotations_view
            size = self.layer._sizes_view

        # if self.layer.dims.ndisplay == 2:
        #     set_data = self.node._subvisuals[self._POINTS_NODE_INDEX].set_data
        # else:
        #     set_data = self.node.set_data

        # set_data(
        #     data[:, ::-1] + 0.5,
        #     size=size,
        #     edge_width=self.layer.edge_width,
        #     symbol=self.layer.symbol,
        #     edge_color=edge_color,
        #     face_color=face_color,
        #     scaling=True,
        # )

        # Update the text
        if self.layer.dims.ndisplay == 2:
            annotation_positions = data + self.layer.annotation_offset
            text_node = self.node._subvisuals[self._TEXT_NODE_INDEX]
            text_node.text = annotations
            # Text axes are flipped
            text_node.pos = np.flip(annotation_positions, axis=1)

        self.node.update()

    def _on_highlight_change(self):
        if self.layer.dims.ndisplay == 3:
            return

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
            annotations = [
                self.layer._annotations_view[i]
                for i in self.layer._highlight_index
            ]

        else:
            data = np.zeros((1, self.layer.dims.ndisplay))
            size = 0
            face_color = 'white'
            annotations = []

        highlight_text_node = self.node._subvisuals[
            self._HIGHLIGHT_TEXT_NODE_INDEX
        ]
        highlight_text_node.text = annotations
        highlight_text_node.pos = np.flip(data, axis=1)
        highlight_text_node.color = self._highlight_color
        highlight_text_node.update()

        if 0 in self.layer._highlight_box.shape:
            pos = np.zeros((1, 2))
            width = 0
        else:
            pos = self.layer._highlight_box
            width = self._highlight_width

        self.node._subvisuals[self._OUTLINE_MARKERS_NODE_INDEX].set_data(
            pos=pos[:, ::-1] + 0.5, color=self._highlight_color, width=width
        )

    def reset(self):
        self._reset_base()
        self._on_data_change()
        self._on_highlight_change()
