import numpy as np
from vispy.scene.visuals import Compound, Line, Rectangle, Text

from napari._vispy.utils.text import get_text_width_height


class ScaleBar(Compound):
    def __init__(self) -> None:
        self._line_data = np.array(
            [
                [-1, 0],
                [1, 0],
                [-1, -1],
                [-1, 1],
                [1, -1],
                [1, 1],
            ]
        )

        self._box_padding = 6
        self._tick_length = 11  # odd numbers look better
        self._color = (1, 1, 1, 1)
        self._box_color = (0, 0, 0, 1)
        self._text_vertices_size = (0, 0)

        self.box = Rectangle(center=[0.5, 0.5], width=100, height=36)
        self.text = Text(
            text='1px',
            pos=[0.5, 0.5],
            anchor_x='center',
            anchor_y='bottom',
            font_size=10,
        )
        self.line = Line(connect='segments', method='gl', width=3)
        # order matters (last is drawn on top)
        super().__init__([self.box, self.text, self.line])

    def set_data(self, *, length, color, ticks, font_size):
        text_width, _ = get_text_width_height(self.text)
        # fixed multiplier for height to avoid fluttering when zooming
        text_height = self.text.font_size * 1.5

        # compute box width and height based on the size of the contents
        box_width = length + self._box_padding * 2
        box_width = max(box_width, text_width + self._box_padding * 2)
        text_line_gap = 5  # gap between text bottom and line top
        box_height = (
            self._tick_length
            + self._box_padding * 2
            + text_height
            + text_line_gap
        )

        line_data = self._line_data if ticks else self._line_data[:2]

        # set the line size based on the length, and position based on
        # the box size and text size with proper spacing
        self.line.set_data(
            pos=line_data * (length / 2, self._tick_length / 2)
            + (
                box_width / 2,
                self._box_padding
                + text_height
                + text_line_gap
                + self._tick_length / 2,
            ),
            color=color,
        )

        self.box.width = box_width
        self.box.height = box_height
        self.box.center = box_width / 2, box_height / 2

        self.text.pos = box_width / 2, self._box_padding
        self.text.color = color
        self.text.font_size = font_size

        # not sure why padding is needed here, ugh
        return box_width, box_height + self._box_padding
