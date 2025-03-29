import numpy as np
from vispy.scene.visuals import Compound, Line, Rectangle, Text


class ScaleBar(Compound):
    def __init__(self) -> None:
        self._data = np.array(
            [
                [0, 0],
                [1, 0],
                [0, -5],
                [0, 5],
                [1, -5],
                [1, 5],
            ]
        )

        # order matters (last is drawn on top)
        super().__init__(
            [
                Rectangle(center=[0.5, 0.5], width=1.1, height=36),
                Text(
                    text='1px',
                    pos=[0.5, 0.5],
                    anchor_x='center',
                    anchor_y='top',
                    font_size=10,
                ),
                Line(connect='segments', method='gl', width=3),
            ]
        )

    @property
    def line(self):
        return self._subvisuals[2]

    @property
    def text(self):
        return self._subvisuals[1]

    @property
    def box(self):
        return self._subvisuals[0]

    def _update_layout(self, font_size):
        # convert font_size to logical pixels as vispy does
        # in vispy/visuals/text/text.py
        # 96 dpi is used as the napari reference dpi
        # round to ensure box.height for font_size 10 is 36
        font_logical_pixels = np.round(font_size * 96 / 72)

        # 18 is the bottom half of the default/initial box
        # 5 is the padding at the top of the text
        self.box.height = 18 + font_logical_pixels + 5

        # Text and line should be fixed at the bottom of the box.
        # At the default font size (10) and box height (36), the position
        # is in the center (0), so subtract half of the default box height (18)
        fixed_position_in_box = self.box.height / 2 - 18
        self.text.pos = [0.5, fixed_position_in_box]
        self._data = np.array(
            [
                [0, fixed_position_in_box],
                [1, fixed_position_in_box],
                [0, fixed_position_in_box - 5],
                [0, fixed_position_in_box + 5],
                [1, fixed_position_in_box - 5],
                [1, fixed_position_in_box + 5],
            ]
        )
        self.line.set_data(pos=self._data)

    def set_data(self, color, ticks):
        data = self._data if ticks else self._data[:2]
        self.line.set_data(data, color)
        self.text.color = color
