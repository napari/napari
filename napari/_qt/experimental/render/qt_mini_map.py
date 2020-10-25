"""MiniMap widget.
"""
import math

import numpy as np
from qtpy.QtGui import QImage, QPixmap
from qtpy.QtWidgets import QLabel

from ....layers.image.experimental.octree_image import OctreeImage

MAP_WIDTH = 100


class MiniMap(QLabel):
    def __init__(self, layer: OctreeImage):
        super().__init__()
        self.layer = layer

    def update(self):
        return

        layer_slice = self.layer._slice
        if layer_slice is None:
            return

        intersection = layer_slice.intersection
        if intersection is None:
            return

        shape = intersection.shape
        ranges = intersection.ranges
        aspect = shape[1] / shape[0]

        map_shape = (MAP_WIDTH, math.ceil(MAP_WIDTH / aspect))
        tile_size = math.ceil(map_shape[1] / shape[1])

        bitmap_shape = map_shape + (4,)
        data = np.zeros(bitmap_shape, dtype=np.uint8)

        def _within(value, value_range):
            return value >= value_range.start and value < value_range.stop

        print("***************MiniMap****************")
        for row in range(0, shape[0]):
            for col in range(0, shape[1]):
                seen = _within(row, ranges[0]) and _within(col, ranges[1])

                y = row * tile_size
                x = col * tile_size
                data[y : y + tile_size, x : x + tile_size] = (
                    (255, 0, 0, 255) if seen else (0, 0, 0, 255)
                )
                mark = "X" if seen else "."
                print(mark, end='')
            print("")

        image = QImage(
            data, data.shape[1], data.shape[0], QImage.Format_RGBA8888,
        )
        self.setPixmap(QPixmap.fromImage(image))
