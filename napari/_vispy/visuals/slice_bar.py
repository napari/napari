from vispy.scene.visuals import Compound, Rectangle, Text


class SliceBar(Compound):
    def __init__(self) -> None:
        # order matters (last is drawn on top)
        super().__init__(
            [
                Rectangle(center=[0.5, 0.5], width=1.1, height=36),
                Text(
                    text='1px',
                    pos=[0.5, 0.5],
                    anchor_x='right',
                    anchor_y='top',
                    font_size=10,
                ),
            ]
        )

    @property
    def text(self):
        return self._subvisuals[1]

    @property
    def box(self):
        return self._subvisuals[0]

    def set_data(self, color):
        self.text.color = color
