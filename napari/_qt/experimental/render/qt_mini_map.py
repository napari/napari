"""MiniMap widget.
"""
import math

import numpy as np
from qtpy.QtGui import QImage, QPixmap
from qtpy.QtWidgets import QLabel

from ....layers.image.experimental.octree_image import OctreeImage
from ....layers.image.experimental.octree_util import OctreeIntersection

MAP_WIDTH = 200

COLOR_SEEN = (255, 0, 0, 255)
COLOR_UNSEEN = (80, 80, 80, 255)
COLOR_VIEW = (227, 220, 111, 255)


class MiniMap(QLabel):
    def __init__(self, viewer, layer: OctreeImage):
        super().__init__()
        self.viewer = viewer
        self.layer = layer

    @property
    def data_corners(self):
        """Return data corners for current view in this layer."""
        # TODO_OCTREE: We need a nice way to access this? Or somehow get the
        # layer to give us the corner_pixels without directly querying the
        # camera.
        qt_viewer = self.viewer.window.qt_viewer
        ndim = self.layer.ndim
        xform = self.layer._transforms[1:].simplified

        corner_pixels = qt_viewer._canvas_corners_in_world[:, -ndim:]
        return xform.inverse(corner_pixels)

    def update(self) -> None:
        """Update the minimap to show latest intersection."""
        intersection = self.layer.get_intersection(self.data_corners)

        if intersection is not None:
            self._draw_map(intersection)

    def _draw_map(self, intersection: OctreeIntersection) -> None:
        """Draw the minimap showing the latest intersection.

        Parameters
        ----------
        intersection : OctreeIntersection
            The intersection we are drawing on the map.
        """
        data = self._get_map_data(intersection)
        height, width = data.shape[:2]
        image = QImage(data, width, height, QImage.Format_RGBA8888)
        self.setPixmap(QPixmap.fromImage(image))

    def _get_map_data(self, intersection: OctreeIntersection) -> np.ndarray:
        """Get the image data to be draw in the map.

        Parameters
        ----------
        intersection : OctreeIntersection
            The intersection we are drawing on the map.
        """
        tile_shape = intersection.info.tile_shape
        aspect = intersection.info.octree_info.aspect

        map_shape = (MAP_WIDTH, math.ceil(MAP_WIDTH / aspect))
        tile_size = math.ceil(map_shape[1] / tile_shape[1])

        bitmap_shape = map_shape + (4,)
        data = np.zeros(bitmap_shape, dtype=np.uint8)

        for row in range(0, tile_shape[0]):
            for col in range(0, tile_shape[1]):
                visible = intersection.is_visible(row, col)

                y0 = row * tile_size + 1
                y1 = y0 + tile_size - 1

                x0 = col * tile_size + 1
                x1 = x0 + tile_size - 1

                data[y0:y1, x0:x1, :] = COLOR_SEEN if visible else COLOR_UNSEEN

        y0, y1 = intersection.normalized_rows * (map_shape[0] - 1)
        x0, x1 = intersection.normalized_cols * (map_shape[1] - 1)

        # TODO_OCTREE: set with one slice operation
        for y in range(int(y0), int(y1)):
            for x in range(int(x0), int(x1)):
                try:
                    data[y, x, :] = COLOR_VIEW
                except IndexError:
                    pass

        return data
