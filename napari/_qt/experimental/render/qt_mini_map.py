"""MiniMap widget.
"""
import math

import numpy as np
from qtpy.QtGui import QImage, QPixmap
from qtpy.QtWidgets import QLabel

from ....components.viewer_model import ViewerModel
from ....layers.image.experimental import OctreeIntersection, OctreeLevel
from ....layers.image.experimental.octree_image import OctreeImage

# Width of the map in the dockable widget.
MAP_WIDTH = 200

# Tiles are seen if they are visible within the current view, otherwise unseen.
COLOR_SEEN = (255, 0, 0, 255)  # red
COLOR_UNSEEN = (80, 80, 80, 255)  # gray

# The view bounds itself is drawn on top of the seen/unseen tiles.
COLOR_VIEW = (227, 220, 111, 255)  # yellow


class MiniMap(QLabel):
    """A small bitmap that shows the view bounds and which tiles are seen.

    Only works with OctreeImage layers.

    Parameters
    ----------
    viewer : Viewer
        The napari viewer.
    layer : OctreeImage
        The octree image we are viewing.
    """

    # Border between the tiles is twice this.
    HALF_BORDER = 1

    def __init__(self, viewer: ViewerModel, layer: OctreeImage):
        super().__init__()
        self.viewer = viewer
        self.layer = layer

    @property
    def data_corners(self):
        """Return data corners for current view in this layer."""
        # TODO_OCTREE: We should not calculate this here. We should query
        # the layer or something to get these corner pixels.
        qt_viewer = self.viewer.window.qt_viewer
        ndim = self.layer.ndim
        xform = self.layer._transforms[1:].simplified

        corner_pixels = qt_viewer._canvas_corners_in_world[:, -ndim:]
        return xform.inverse(corner_pixels)

    def update(self) -> None:
        """Update the minimap to show latest intersection."""
        # This actually performs the intersection, but it's very fast.
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
        data = self._create_map_data(intersection)
        height, width = data.shape[:2]
        image = QImage(data, width, height, QImage.Format_RGBA8888)
        self.setPixmap(QPixmap.fromImage(image))

    def _create_map_data(self, intersection: OctreeIntersection) -> np.ndarray:
        """Return bitmap data for the map.

        Draw the tiles and the intersection with those tiles.

        Parameters
        ----------
        intersection : OctreeIntersection
            Draw this intersection on the map.
        """
        aspect = intersection.level.info.octree_info.aspect

        # Shape of the map bitmap: (row_pixels, col_pixels)
        map_shape = math.ceil(MAP_WIDTH / aspect), MAP_WIDTH

        # The map shape with RGBA pixels
        bitmap_shape = map_shape + (4,)

        # The bitmap data.
        data = np.zeros(bitmap_shape, dtype=np.uint8)

        # Leave a bit of space between the tiles.
        edge = self.HALF_BORDER

        level: OctreeLevel = intersection.level

        scale_x = map_shape[1] / level.info.image_shape[1]
        scale_y = map_shape[0] / level.info.image_shape[0]

        # OCTREE_TODO: Consider this Qt bitmap rendering code just a rough
        # proof-of-concept prototype. A real minimap should probably be a
        # little OpenGL window, not a Qt bitmap. If we used OpenGL the GPU
        # does most of the work and we get better quality and effects, like
        # alpha and anti-aliasing. We do not want to spend a lot of time
        # becoming good at "rendering" into Qt Bitmaps.
        y = 0
        for row, row_tiles in enumerate(level.tiles):
            x = 0
            for col, tile in enumerate(row_tiles):
                # Is this tile in the view?
                seen = intersection.is_visible(row, col)

                tile_x = tile.shape[1] * scale_x
                tile_y = tile.shape[0] * scale_y

                y0 = int(y)
                x0 = int(x)

                y1 = int(y + tile_y)
                x1 = int(x + tile_x)

                # Create a small border between tiles.
                y0 += edge
                y1 -= edge
                x0 += edge
                x1 -= edge

                # print(f"tile {row}, {col}: {y0}:{y1}  {x0}:{x1}")

                # Draw one tile.
                data[y0:y1, x0:x1, :] = COLOR_SEEN if seen else COLOR_UNSEEN

                x += tile_x
            y += tile_y

        self._draw_view(data, intersection)
        return data

    def _draw_view(self, data, intersection: OctreeIntersection) -> None:
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
