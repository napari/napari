"""QtMiniMap widget.

Creates a bitmap that shows which octree tiles are being viewed.
"""
import math
from typing import NamedTuple

import numpy as np
from qtpy.QtGui import QImage, QPixmap
from qtpy.QtWidgets import QLabel

from ....layers.image.experimental import OctreeIntersection, OctreeLevel
from ....layers.image.experimental.octree_image import OctreeImage

# Longest edge of map bitmap in pixels. So if in a odd shape it does not
# become bigger than this in either direction.
MAP_SIZE = 220

# Seen tiles are inside the view.
COLOR_SEEN = (255, 0, 0, 255)  # red
COLOR_UNSEEN = (80, 80, 80, 255)  # gray

# The view bounds are draw on top of the seen/unseen tiles.
COLOR_VIEW = (227, 220, 111, 255)  # yellow

# Create a gap between the tiles.
TILE_EDGE = 1


class Rect(NamedTuple):
    """Rectangle that we can draw into a numpy array."""

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


def _draw_tiles(
    bitmap: np.ndarray, intersection: OctreeIntersection, scale: np.ndarray
) -> None:
    """Draw all the tiles, marking which are seen by the intersection.

    Parameters
    ----------
    intersection : OctreeIntersection
        The intersection we are drawing.
    scale_xy : Tuple[float, float]
        The scale to draw things that.
    """
    level = intersection.level
    shape = level.info.slice_config.base_shape

    y = 0
    for row in range(shape[0]):
        x = 0
        for col in range(shape[1]):
            octree_chunk = level.get_chunk(row, col)

            if octree_chunk is None:
                print(f"NO CHUNK: {row}, {col}")
                scaled = [0, 0]
                continue  # should not happen?

            scaled = octree_chunk.data.shape[:2] * scale
            rect = Rect(x, y, scaled[1], scaled[0])
            row, col = octree_chunk.location.row, octree_chunk.location.col

            color = (
                COLOR_SEEN
                if intersection.is_visible(row, col)
                else COLOR_UNSEEN
            )

            rect.draw(bitmap, color)
            x += scaled[1]
        y += scaled[0]


def _get_bitmap_shape(aspect: float) -> np.ndarray:
    """Get shape for the map bitmap.

    Parameters
    ----------
    aspect : float
        The width:height aspect ratio of the base image for the map.

    Return
    ------
    Tuple[int, int]
        The shape for the map bitmap.
    """
    depth = 4  # RGBA

    # Limit to at most MAP_SIZE pixels, in whichever dimension is the
    # bigger one. So it's not too huge even if an odd shape.
    if aspect > 1:
        # Limit the width, it is longer.
        return np.array((math.ceil(MAP_SIZE / aspect), MAP_SIZE, depth))

    # Limit the height, it is longer.
    return np.array((MAP_SIZE, math.ceil(MAP_SIZE * aspect), depth))


def _draw_intersection(intersection: OctreeIntersection) -> np.ndarray:
    """Return a bitmap that shows the tiles and the intersection.

    Parameters
    ----------
    intersection : OctreeIntersection
        Draw this intersection.

    Returns
    -------
    np.ndarray
        The bitmap showing the intersection.
    """
    aspect = intersection.level.info.slice_config.aspect_ratio
    level: OctreeLevel = intersection.level

    # Map shape plus RGBA depth.
    bitmap_shape = _get_bitmap_shape(aspect)
    data = np.zeros(bitmap_shape, dtype=np.uint8)

    # Scale the intersection down to fit in the bitmap.
    scale = bitmap_shape[:2] / level.info.image_shape

    # Draw all the tiles, the seen ones are drawn in red.
    _draw_tiles(data, intersection, scale)

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
        intersection = self.layer.get_intersection()

        if intersection is not None:
            self._draw(intersection)

    def _draw(self, intersection: OctreeIntersection) -> None:
        """Draw the minimap showing the latest intersection.

        Update us with the new bitmap, we are a QLabel.

        Parameters
        ----------
        intersection : OctreeIntersection
            The intersection we are drawing on the mini map.
        """
        data = _draw_intersection(intersection)
        height, width = data.shape[:2]
        image = QImage(data, width, height, QImage.Format_RGBA8888)
        self.setPixmap(QPixmap.fromImage(image))
