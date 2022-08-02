import numpy as np
from vispy.scene.visuals import Line, Rectangle, Text
from vispy.visuals.transforms import STTransform


class ScaleBar(Line):
    def __init__(self):
        self._data = np.array(
            [
                [0, 0, -1],
                [1, 0, -1],
                [0, -5, -1],
                [0, 5, -1],
                [1, -5, -1],
                [1, 5, -1],
            ]
        )

        self.text = Text(pos=[0.5, -1])
        self.box = Rectangle(center=[0.5, 0.5], width=1.1, height=36)

        super().__init__(connect='segments', method='gl', width=3)

        self.text.parent = self
        self.text.order = 0
        self.text.transform = STTransform(translate=(0, 10, 0, 0))
        self.text.font_size = 10
        self.text.anchors = ("center", "center")
        self.text.text = "1px"

        self.box.parent = self
        self.box.order = 1
        self.box.transform = STTransform(translate=(0, 20, 0, 0))

    def set_data(self, color, ticks):
        data = self._data if ticks else self._data[:2]
        super().set_data(data, color)
        self.text.color = color
