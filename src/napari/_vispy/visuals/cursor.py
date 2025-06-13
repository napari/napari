import numpy as np
from vispy.scene import Line


class Cursor(Line):
    _segments = (
        np.array(
            [
                [0, 0, 1],
                [0, 0, -1],
                [0, 1, 0],
                [0, -1, 0],
                [1, 0, 0],
                [-1, 0, 0],
            ]
        )
        * 1e6
    )
    # TODO: for some reason higher numbers here break in 3D...

    def __init__(self):
        super().__init__(self._segments, connect='segments', color='red')

    def set_position(self, value: np.ndarray) -> None:
        self.set_data(pos=self._segments + value)
