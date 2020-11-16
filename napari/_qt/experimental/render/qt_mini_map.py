"""QtMiniMap widget.
"""
import math
from typing import NamedTuple

import numpy as np
from qtpy.QtGui import QImage, QPixmap
from qtpy.QtWidgets import QLabel

from ....layers.image.experimental import OctreeIntersection, OctreeLevel
from ....layers.image.experimental.octree_image import OctreeImage

# Longest edge of map bitmap in pixels. So at most MAP_SIZE wide and at
# most MAP_SIZE high. In case it's narrow one way or the other.
MAP_SIZE = 220

# Tiles are seen if they are visible within the current view, otherwise unseen.
COLOR_SEEN = (255, 0, 0, 255)  # red
COLOR_UNSEEN = (80, 80, 80, 255)  # gray

# The view bounds itself is drawn on top of the seen/unseen tiles.
COLOR_VIEW = (227, 220, 111, 255)  # yellow

# Edge around tiles, so gap is twice this.
TILE_EDGE = 1


class Rect(NamedTuple):
    """Rectangle that we can "draw" into a numpy array."""

    x: float
    y: float
    width: float
    height: float

    def draw(self, data: np.ndarray, color: np.ndarray) -> None:
        """Draw the rectangle into the array."""
        y0, y1 = int(self.y), int(self.y + self.height)
        x0, x1 = int(self.x), int(self.x + self.width)

        # Create a small border between tiles.
        y0 += TILE_EDGE
        y1 -= TILE_EDGE
        x0 += TILE_EDGE
        x1 -= TILE_EDGE

        data[y0:y1, x0:x1, :] = color


def _draw_view(data, intersection: OctreeIntersection) -> None:
    """Draw the view rectangle onto the map data.

    Parameters
    ----------
    data : np.ndarray
        Draw the view into this data.
    intersection : OctreeIntersection
        Draw the view in this intersection.
    """
    # Max (row, col) dimensions of the bitmap we are writing into.
    max_dim = np.array([data.shape[0] - 1, data.shape[1] - 1])

    # Convert normalized ranges into bitmap pixel ranges
    ranges = (intersection.normalized_range * max_dim).astype(int)

    # Write the view color into this rectangular regions.
    # TODO_OCTREE: must be a nicer way to index this?
    rows, cols = ranges[0], ranges[1]
    data[rows[0] : rows[1], cols[0] : cols[1], :] = COLOR_VIEW


def _draw_tiles(data, intersection, level, scale_xy) -> None:

    y = 0
    for row, row_tiles in enumerate(level.tiles):
        x = 0
        for col, tile in enumerate(row_tiles):

            tile_x = tile.shape[1] * scale_xy[0]
            tile_y = tile.shape[0] * scale_xy[1]

            rect = Rect(x, y, tile_x, tile_y)

            color = (
                COLOR_SEEN
                if intersection.is_visible(row, col)
                else COLOR_UNSEEN
            )

            rect.draw(data, color)

            x += tile_x
        y += tile_y


def _create_map_data(intersection: OctreeIntersection) -> np.ndarray:
    """Return bitmap data for the map.

    Draw the tiles and the intersection with those tiles.

    Parameters
    ----------
    intersection : OctreeIntersection
        Draw this intersection on the map.
    """
    aspect = intersection.level.info.image_config.aspect

    # Limit to at most MAP_SIZE pixels, in whichever dimension is the
    # bigger one. So it's not too huge even if an odd shape.
    if aspect > 1:
        map_shape = math.ceil(MAP_SIZE / aspect), MAP_SIZE
    else:
        map_shape = MAP_SIZE, math.ceil(MAP_SIZE * aspect)

    # Shape of the map bitmap: (row_pixels, col_pixels)

    # The map shape with RGBA pixels
    bitmap_shape = map_shape + (4,)

    # The bitmap data.
    data = np.zeros(bitmap_shape, dtype=np.uint8)

    level: OctreeLevel = intersection.level

    scale_xy = [
        map_shape[1] / level.info.image_shape[1],
        map_shape[0] / level.info.image_shape[0],
    ]

    # Draw all the tiles, the seen ones in red.
    _draw_tiles(data, intersection, level, scale_xy)

    # Draw the view frustum in yellow.
    _draw_view(data, intersection)

    return data


class QtMiniMap(QLabel):
    """A small bitmap that shows the view bounds and which tiles are seen.

    Only works with OctreeImage layers.

    TODO_OCTREE: Consider this Qt bitmap rendering code just a rough
    proof-of-concept prototype. A real minimap should probably be a little
    OpenGL window, not a Qt bitmap. If we used OpenGL the GPU does most of
    the work and we get better quality and effects, like alpha and
    anti-aliasing. We do not want to spend a lot of time becoming good at
    "rendering" into Qt Bitmaps.

    Parameters
    ----------
    layer : OctreeImage
        The octree image we are viewing.
    """

    def __init__(self, layer: OctreeImage):
        super().__init__()
        self.layer = layer

    def update(self) -> None:
        """Update the minimap to show latest intersection."""
        # This actually performs the intersection, but it's very fast.
        intersection = self.layer.get_intersection()

        if intersection is not None:
            self._draw_map(intersection)

    def _draw_map(self, intersection: OctreeIntersection) -> None:
        """Draw the minimap showing the latest intersection.

        Parameters
        ----------
        intersection : OctreeIntersection
            The intersection we are drawing on the map.
        """
        data = _create_map_data(intersection)
        height, width = data.shape[:2]
        image = QImage(data, width, height, QImage.Format_RGBA8888)
        self.setPixmap(QPixmap.fromImage(image))
