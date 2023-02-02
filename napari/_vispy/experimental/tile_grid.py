"""TileGrid class.

A grid drawn around/between the tiles for debugging and demos.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, List

import numpy as np
from vispy.scene.node import Node
from vispy.scene.visuals import Line

if TYPE_CHECKING:
    from napari.layers.image.experimental import OctreeChunk

# Grid lines drawn with this width and color.
GRID_WIDTH = 3
GRID_COLOR = (1, 0, 0, 1)

# Draw grid on top of the tiles.
LINE_VISUAL_ORDER = 10


# Outline for 'segments' points, each pair is one line segment.
_OUTLINE = np.array(
    [[0, 0], [1, 0], [1, 0], [1, 1], [1, 1], [0, 1], [0, 1], [0, 0]],
    dtype=np.float32,
)


def _chunk_outline(chunk: OctreeChunk) -> np.ndarray:
    """Return the verts that outline this single chunk.

    The Line is should be drawn in 'segments' mode.

    Parameters
    ----------
    chunk : OctreeChunk
        Create outline of this chunk.

    Returns
    -------
    np.ndarray
        The verts for the outline.
    """
    geom = chunk.geom
    x, y = geom.pos
    w, h = geom.size

    outline = _OUTLINE.copy()  # Copy and modify in place.
    outline[:, :2] *= (w, h)
    outline[:, :2] += (x, y)

    return outline


class TileGrid:
    """A grid to show the outline of all the tiles.

    Created for debugging and demos, but we might show for real in certain
    situations, like while the tiles are loading?

    Attributes
    ----------
    parent : Node
        The parent of the grid.
    """

    def __init__(self, parent: Node) -> None:
        self.parent = parent
        self.line = self._create_line()

    def _create_line(self) -> Line:
        """Create the Line visual for the grid.

        Returns
        -------
        Line
            The new Line visual.
        """
        line = Line(connect='segments', color=GRID_COLOR, width=GRID_WIDTH)
        line.order = LINE_VISUAL_ORDER
        line.parent = self.parent
        return line

    def update_grid(self, chunks: List[OctreeChunk], base_shape=None) -> None:
        """Update grid for this set of chunks and the whole boundary.

        Parameters
        ----------
        chunks : List[ImageChunks]
            Add a grid that outlines these chunks.
        base_shape : List[int], optional
            Height and width of the full resolution level.
        """
        verts = np.zeros((0, 2), dtype=np.float32)
        for octree_chunk in chunks:
            chunk_verts = _chunk_outline(octree_chunk)
            verts = np.vstack([verts, chunk_verts])

        # Add in the base shape outline if provided
        if base_shape is not None:
            outline = _OUTLINE.copy()  # Copy and modify in place.
            outline[:, :2] *= base_shape[::-1]
            verts = np.vstack([verts, outline])

        self.line.set_data(verts)

    def clear(self) -> None:
        """Clear the grid so nothing is drawn."""
        data = np.zeros((0, 2), dtype=np.float32)
        self.line.set_data(data)
