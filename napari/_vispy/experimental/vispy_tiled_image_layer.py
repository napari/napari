"""VispyTiledImageLayer class.
"""
from typing import List

import numpy as np
from vispy.scene.visuals import Line
from vispy.visuals.transforms import STTransform

from ...layers.image.experimental.octree_util import ChunkData
from ...utils.perf import block_timer
from ..image import Image as ImageNode
from ..vispy_image_layer import VispyImageLayer

GRID_WIDTH = 3
GRID_COLOR = (1, 0, 0, 1)

INITIAL_POOL_SIZE = 100


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

    def __init__(self, chunk_data: ChunkData, node=None):
        self.chunk_data = chunk_data
        self.data_id = id(chunk_data.data)
        if node is None:
            self.node = ImageNode(None, method='auto')
        else:
            self.node = node
        self.node.order = 0


class ImageNodePool:
    def __init__(self):
        size = INITIAL_POOL_SIZE
        with block_timer(
            f"ImageNodePool: create pool size {size}", print_time=True
        ):
            self.nodes = self._create_pool(size)
        self.index = 0

    def _create_pool(self, size):
        return [ImageNode(None, method='auto') for x in range(size)]

    def get_node(self):
        node = self.nodes[self.index]
        self.index += 1
        return node


class TileGrid:
    """The grid that shows the outlines of all the tiles."""

    def __init__(self, parent):
        self.parent = parent
        self.line = self._create_line()

    def _create_line(self):
        line = Line(connect='segments', color=GRID_COLOR, width=GRID_WIDTH)
        line.order = 10
        line.parent = self.parent
        return line

    def add_chunks(self, chunks: List[ImageChunk]) -> None:
        """Add the outline of the given chunks to the grid.

        Parameters
        ----------
        chunks : List[ImageChunks]
            Add the outline of these chunks
        """
        # TODO_OCTREE: create in one go without vstack
        verts = np.zeros((0, 2), dtype=np.float32)
        for image_chunk in chunks:
            chunk_verts = _chunk_outline(image_chunk.chunk_data)
            verts = np.vstack([verts, chunk_verts])

        self.line.set_data(verts)
        self.verts = verts


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

    def __init__(self, layer, camera):
        self.ready = False
        self.camera = camera
        self.image_chunks = {}
        self.grid = None  # Can't create until after super() init
        self.pool = ImageNodePool()

        # This will call our self._on_data_change() but we guard
        # that with self.read as a hack.
        super().__init__(layer)

        self.grid = TileGrid(self.node)
        self.camera.events.center.connect(self._on_camera_move)
        self.ready = True

    def _create_image_chunk(self, chunk_data: ChunkData):
        """Create a new ImageChunk object.

        Parameters
        ----------
        chunk_data : ChunkData
            The data used to create the new image chunk.
        """
        node = self.pool.get_node()
        image_chunk = ImageChunk(chunk_data, node)

        # Parent VispyImageLayer will process the data then set it.
        self._set_node_data(image_chunk.node, chunk_data.data)

        # Add this new ImageChunk as child of self.node, transformed into place.
        image_chunk.node.parent = self.node
        image_chunk.node.transform = STTransform(
            chunk_data.scale, chunk_data.pos
        )

        return image_chunk

    def _on_data_change(self, event=None) -> None:
        """Our self.layer._data_view has been updated, update our node.
        """
        if not self.layer.loaded or not self.ready:
            return

        self._update_visible_chunks()

    def _update_visible_chunks(self) -> None:

        # Get the currently visible chunks.
        visible_chunks = self.layer.visible_chunks

        # Data id's of the visible chunks
        visible_ids = set([id(chunk.data) for chunk in visible_chunks])

        num_start = len(self.image_chunks)

        # Create new chunks.
        for chunk in visible_chunks:
            chunk_id = id(chunk.data)
            if chunk_id not in self.image_chunks:
                self.image_chunks[chunk_id] = self._create_image_chunk(chunk)
                break  # testing: one at a time

        num_peak = len(self.image_chunks)
        num_created = num_peak - num_start

        # Remove stale chunks.
        for image_chunk in list(self.image_chunks.values()):
            data_id = image_chunk.data_id
            if data_id not in visible_ids:
                image_chunk.node.parent = None
                del self.image_chunks[data_id]

        num_final = len(self.image_chunks)
        num_deleted = num_peak - num_final

        self.grid.add_chunks(self.image_chunks.values())

        if num_created > 0 or num_deleted > 0:
            print(
                f"VispyTiled: start: {num_start} created: {num_created} "
                f"deleted: {num_deleted} final: {num_final}"
            )

    def _on_camera_move(self, event=None):
        self._update_visible_chunks()
