import numpy as np
from vispy.scene.visuals import Compound, Line

from napari._vispy.visuals.text import Text


class ScaleBar(Compound):
    """Scale bar visual with text and line components.

    Layout of Scale Bar Elements (from top to bottom):
    - Padding
    - Text
    - Gap between text and line
    - Scale line (with optional ticks)
    - Padding
    """

    def __init__(self) -> None:
        # Layout constants
        self.PADDING = 6  # Space around the entire scale bar
        self.TICK_LENGTH = 11  # Height of tick marks (odd numbers look better)

        # Line geometry: main line + optional tick marks
        self._line_data = np.array(
            [
                [-1, 0],  # Left end of main line
                [1, 0],  # Right end of main line
                [-1, -1],  # Left tick mark (bottom)
                [-1, 1],  # Left tick mark (top)
                [1, -1],  # Right tick mark (bottom)
                [1, 1],  # Right tick mark (top)
            ]
        )

        self._color = (1, 1, 1, 1)

        self.text = Text(
            text='1px',
            pos=[0.5, 0.5],
            anchor_x='center',
            anchor_y='bottom',
            font_size=10,
        )
        self.line = Line(
            connect='segments', method='gl', width=3, antialias=True
        )
        # order matters (last is drawn on top)
        super().__init__([self.text, self.line])

    def _calculate_layout(self, length: float) -> dict:
        """Calculate all layout dimensions and positions."""
        # Text dimensions
        text_width, text_height = self.text.get_width_height()

        # ceil so there's no flickering due to slight differences between
        # the heigh of the various digits
        text_height = np.ceil(text_height)

        width = max(
            length + self.line.width,
            text_width,
        )
        height = text_height + self.TICK_LENGTH

        line_center_y = text_height + (self.TICK_LENGTH / 2)

        return {
            'width': width,
            'height': height,
            'line_center_y': line_center_y,
        }

    def set_data(self, *, length, color, ticks, font_size):
        """Update scale bar with new dimensions and styling."""
        # font size need to be set first cause layout calculations depend on it
        self.text.font_size = font_size

        layout = self._calculate_layout(length)

        # Choose line data based on whether ticks are enabled
        line_data = self._line_data if ticks else self._line_data[:2]

        # Position and scale the line
        self.line.set_data(
            pos=line_data * (length / 2, self.TICK_LENGTH / 2)
            + (
                layout['width'] / 2,  # Center horizontally
                layout['line_center_y'],  # Position vertically
            ),
            color=color,
        )

        # Position the text
        self.text.pos = layout['width'] / 2, 0
        self.text.color = color

        # Return dimensions for the overlay system
        # Extra padding needed for proper canvas positioning (not sure why padding is needed here, ugh)
        return layout['width'], layout['height']
