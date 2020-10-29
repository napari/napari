"""VispyTiledImageLayer class.
"""
import numpy as np
from vispy.scene.visuals import Line
from vispy.visuals.transforms import STTransform

from ...layers.image.experimental.octree_util import ChunkData
from ..image import Image as ImageNode
from ..vispy_image_layer import VispyImageLayer

GRID_WIDTH = 3
GRID_COLOR = (1, 0, 0, 1)


def _chunk_outline(chunk: ChunkData) -> np.ndarray:
    """Return the line verts that outline the given chunk.

    Parameters
    ----------
    chunk : ChunkData
        Create outline of this chunk.

    Return
    ------
    np.ndarray
        The chunk verts for 'segments' mode.
    """
    x, y = chunk.pos
    h, w = chunk.data.shape[:2]
    w *= chunk.scale[1]
    h *= chunk.scale[0]

    # We draw lines on all four sides of the chunk. This means are
    # double-drawing all interior lines in the grid. We can draw less if
    # performance is an issue.
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


class ImageChunk:
    """The ImageNode for a single chunk.

    This class will grow soon...
    """

    def __init__(self):
        self.node = ImageNode(None, method='auto')
        self.node.order = 0


class TileGrid:
    """The grid that shows the outlines of all the tiles."""

    def __init__(self):
        self.reset()
        self.line = Line(
            connect='segments', color=GRID_COLOR, width=GRID_WIDTH
        )
        self.line.order = 10

    def reset(self) -> None:
        """Reset the grid to have no lines."""
        self.verts = np.zeros((0, 2), dtype=np.float32)

    def add_chunk(self, chunk: ChunkData) -> None:
        """Add the outline of the given chunk to the grid.

        Parameters
        ----------
        chunk : ChunkData
            Add the outline of this chunk.
        """
        chunk_verts = _chunk_outline(chunk)

        # Clear verts first. Prevents segfault when we modify self.verts.
        self.line.set_data([])

        self.verts = np.vstack([self.verts, chunk_verts])
        self.line.set_data(self.verts)


class VispyTiledImageLayer(VispyImageLayer):
    """Tiled images using multiple ImageNodes.

    Render a set of image tiles. For example render the set of tiles that
    need to be drawn from an octree in order to depict the current view.

    We create a parent ImageNode, which is current empty. Then we create a
    child ImageNode for every image tile. The set of child ImageNodes will
    change over time as the user pans/zooms in the octree.

    Future Work
    -----------

    This class will likely need to be replaced at some point. We plan to
    write a TiledImageVisual class which stores the tile textures in an
    atlas-like tile cache. Then in one draw command it can draw the all the
    visible tiles in the right positions using texture coordinates.

    One reason TiledImageVisual will be faster is today calling
    ImageNode.set_data() causes a 15-20ms hiccup when that visual is next
    drawn. It's not clear why the hiccup is so big, since transfering the
    texture into VRAM should be under 2ms. However at least 4ms is spent
    rebuilding the shader alone.

    Since we might need to draw 10+ new tiles at one time, the total delay
    could be up to 200ms total, which is too slow.

    However, this initial approach with separate ImageVisuals should be a
    good starting point, since it will let us iron out the octree
    implementation and the tile placement math. Then we can upgrade
    to the new TiledImageVisual when its ready, and everything should
    still look the same, it will just be faster.
    """

    def __init__(self, layer):
        self.chunks = {}
        self.grid = TileGrid()

        # This will call our self._on_data_change().
        super().__init__(layer)

        self.grid.line.parent = self.node

    def _create_image_chunk(self, chunk: ChunkData):
        """Create a new ImageChunk object.

        Parameters
        ----------
        chunk : ChunkData
            The data used to create the new image chunk.
        """
        image_chunk = ImageChunk()

        self.grid.add_chunk(chunk)

        # Parent VispyImageLayer will process the data then set it.
        self._set_node_data(image_chunk.node, chunk.data)

        # Add this new ImageChunk as child of self.node, transformed into place.
        image_chunk.node.parent = self.node
        image_chunk.node.transform = STTransform(chunk.scale, chunk.pos)

        return image_chunk

    def _on_data_change(self, event=None) -> None:
        """Our self.layer._data_view has been updated, update our node.
        """
        if not self.layer.loaded:
            # Do nothing if we are not yet loaded.
            return

        print(f"VispyTiled: delete {len(self.chunks)} chunks")

        # For now, nuke all the old chunks. Soon we will keep the ones
        # which are still being drawn.
        for image_chunk in self.chunks.values():
            image_chunk.node.parent = None
        self.chunks = {}
        self.grid.reset()

        chunks = self.layer.view_chunks

        print(f"VispyTiled: create {len(chunks)} chunks")

        # Create the new chunks.
        for chunk in chunks:
            chunk_id = id(chunk.data)
            if chunk_id not in self.chunks:
                self.chunks[chunk_id] = self._create_image_chunk(chunk)
