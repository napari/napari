import numpy as np
from vispy.scene.visuals import Line, Compound
from vispy.scene.visuals import Text as TextNode
from vispy.visuals.transforms import ChainTransform

from .vispy_base_layer import VispyBaseLayer


class VispyTextLayer(VispyBaseLayer):
    _highlight_color = (0, 0.6, 1)
    _highlight_width = 1.5

    _OUTLINE_MARKERS_NODE_INDEX = 0
    _HIGHLIGHT_TEXT_NODE_INDEX = 2

    def __init__(self, layer):
        # Create a compound visual with the following thress subvisuals:
        # Lines: The lines of the interaction box used for highlights.
        # Text: The text to be diplayed.
        # Text: Highlighted text to show which text is selected.
        node = Compound([Line(), TextNode(), TextNode()])
        self._text_node_index = 1

        super().__init__(layer, node)

        self.layer.events.rotation.connect(lambda e: self._on_data_change())
        self.layer.events.font_size.connect(lambda e: self._on_data_change())
        self.layer.events.text_color.connect(lambda e: self._on_data_change())
        self.layer.events.highlight.connect(
            lambda e: self._on_highlight_change()
        )

        self._on_display_change()
        self._on_data_change()

    def _on_display_change(self):
        parent = self.node.parent
        self.node.transforms = ChainTransform()
        self.node.parent = None

        if self.layer.dims.ndisplay == 2:
            self.node = Compound([Line(), TextNode(), TextNode()])
            self._text_node_index = 1
        else:
            self.node = TextNode()
            self._text_node_index = 0

        self.node.parent = parent
        self._reset_base()
        self.layer._update_dims()
        self.layer._set_view_slice()

    def _on_data_change(self):
        # Set vispy data, noting that the order of the text needs to be
        # reversed to make the most recently added text appear on top
        # and the rows / columns need to be switch for vispys x / y ordering

        # Check if ndisplay has changed current node type needs updating
        if (
            self.layer.dims.ndisplay == 3
            and not isinstance(self.node, TextNode)
        ) or (
            self.layer.dims.ndisplay == 2
            and not isinstance(self.node, Compound)
        ):
            self._on_display_change()
            self._on_highlight_change()

        if len(self.layer._text_coords_view) == 0:
            coords = np.zeros((1, self.layer.dims.ndisplay))
            text = []

        else:
            coords = self.layer._text_coords_view
            text = self.layer._text_view

        if isinstance(self.node, TextNode):
            text_node = self.node
        else:
            text_node = self.node._subvisuals[self._text_node_index]
        # Update the text
        if self.layer.dims.ndisplay == 2:
            positions = np.flip(coords, axis=1)
        elif self.layer.dims.ndisplay == 3:
            raw_positions = np.flip(coords, axis=1)
            n_positions, position_dims = raw_positions.shape

            if position_dims < 3:
                padded_positions = np.zeros((n_positions, 3))
                padded_positions[:, 0:2] = raw_positions
                positions = padded_positions
            else:
                positions = raw_positions

        self._update_text_node(
            text_node,
            text=text,
            pos=positions,
            rotation=self.layer.rotation,
            color=self.layer.text_color,
            font_size=self.layer.font_size,
        )

    def _on_highlight_change(self):
        if self.layer.dims.ndisplay == 3:
            return

        if len(self.layer._highlight_index) > 0:
            # Color the hovered or selected text
            coords = self.layer._text_coords_view[self.layer._highlight_index]
            if coords.ndim == 1:
                coords = np.expand_dims(coords, axis=0)
            text = [
                self.layer._text_view[i] for i in self.layer._highlight_index
            ]

        else:
            coords = np.zeros((1, self.layer.dims.ndisplay))
            text = []

        highlight_text_node = self.node._subvisuals[
            self._HIGHLIGHT_TEXT_NODE_INDEX
        ]
        self._update_text_node(
            highlight_text_node,
            text=text,
            pos=np.flip(coords, axis=1),
            rotation=self.layer.rotation,
            color=self._highlight_color,
            font_size=self.layer.font_size,
        )

        if 0 in self.layer._highlight_box.shape:
            pos = np.zeros((1, 2))
            width = 0
        else:
            pos = self.layer._highlight_box
            width = self._highlight_width

        self.node._subvisuals[self._OUTLINE_MARKERS_NODE_INDEX].set_data(
            pos=pos[:, ::-1] + 0.5, color=self._highlight_color, width=width
        )

    def _update_text_node(
        self, node, text=[], rotation=0, color='black', font_size=12, pos=None
    ):
        node.text = text
        node.pos = pos
        node.rotation = rotation
        node.color = color
        node.font_size = font_size

        node.update()
