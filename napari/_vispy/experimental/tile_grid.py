"""TileGrid class.
"""
from typing import List

import numpy as np
from vispy.scene.node import Node
from vispy.scene.visuals import Line

from ...layers.image.experimental.octree_util import ChunkData

# Draw with lines of this width and color.
GRID_WIDTH = 3
GRID_COLOR = (1, 0, 0, 1)

# Draw on tope of the tiles.
LINE_VISUAL_ORDER = 10


def _chunk_outline(chunk: ChunkData) -> np.ndarray:
    """Return the verts that outline this single chunk.

    The Line is should be drawn in 'segments' mode.

    Parameters
    ----------
    chunk : ChunkData
        Create outline of this chunk.

    Return
    ------
    np.ndarray
        The verts for the outline.
    """
    x, y = chunk.pos
    h, w = chunk.data.shape[:2]
    w *= chunk.scale[1]
    h *= chunk.scale[0]

    # We draw lines on all four sides of the chunk. This means are
    # double-drawing all interior lines in the grid. We can avoid
    # this duplication if it becomes a performance issue.

    # TODO_OCTREE: use */+ like we do with quads
    return np.array(
        (
            [x, y],
            [x + w, y],
            [x + w, y],
            [x + w, y + h],
            [x + w, y + h],
            [x, y + h],
            [x, y + h],
            [x, y],
        ),
        dtype=np.float32,
    )


class TileGrid:
    """A grid to show the outline of all the tiles.

    Created for debugging although could be shown for real as well.

    Attributes
    ----------
    parent : Node
        The parent of the grid.
    """

    def __init__(self, parent: Node):
        self.parent = parent
        self.line = self._create_line()

    def _create_line(self) -> Line:
        """Create the Line visual for the grid.

        Return
        ------
        Line
            The new Line visual.
        """
        line = Line(connect='segments', color=GRID_COLOR, width=GRID_WIDTH)
        line.order = LINE_VISUAL_ORDER
        line.parent = self.parent
        return line

    def update_grid(self, chunks: List[ChunkData]) -> None:
        """Update grid for this set of chunks.

        Parameters
        ----------
        chunks : List[ImageChunks]
            Add a grid that outlines these chunks.
        """
        # TODO_OCTREE: create in one go without vstack?
        verts = np.zeros((0, 2), dtype=np.float32)
        for chunk_data in chunks:
            chunk_verts = _chunk_outline(chunk_data)
            verts = np.vstack([verts, chunk_verts])

        self.line.set_data(verts)

    def clear(self) -> None:
        """Clear the grid so nothing is drawn."""
        data = np.zeros((0, 2), dtype=np.float32)
        self.line.set_data(data)
