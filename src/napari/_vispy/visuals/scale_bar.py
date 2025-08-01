import numpy as np
from vispy.scene.visuals import Compound, Line, Rectangle, Text


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

    def set_data(self, length, color, ticks, font_size):
        if self.text.transforms.dpi:
            # use 96 as the napari reference dpi for historical reasons
            dpi_scale_factor = 96 / self.text.transforms.dpi
        else:
            dpi_scale_factor = 1

        font_size *= dpi_scale_factor

        vert_buffer = self.text._vertices_data
        if vert_buffer is not None:
            pos = vert_buffer['a_position']
            tl = pos.min(axis=0)
            br = pos.max(axis=0)
            self._text_vertices_size = (br[0] - tl[0]), (br[1] - tl[1])

        text_width = self._text_vertices_size[0] * font_size * 1.3  # magic?
        # fixed multiplier for height to avoid fluttering when zooming
        text_height = font_size * 1.5

        box_width = length + self._box_padding * 2
        box_width = max(box_width, text_width + self._box_padding * 2)
        box_height = (
            self._tick_length / 2 + self._box_padding * 2 + text_height
        )

        line_data = self._line_data if ticks else self._line_data[:2]
        self.line.set_data(
            line_data * (length / 2, self._tick_length / 2)
            + (
                box_width / 2,
                self._box_padding + text_height,
            ),
            color,
        )

        self.box.width = box_width
        self.box.height = box_height
        self.box.center = box_width / 2, box_height / 2

        self.text.pos = box_width / 2, self._box_padding
        self.text.color = color
        self.text.font_size = font_size

        # not sure why padding is needed here, ugh
        return box_width, box_height + self._box_padding
