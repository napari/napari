"""VispyTiledImageLayer class.
"""
from typing import List, Optional, Set

import numpy as np
from vispy.scene.node import Node
from vispy.scene.visuals import Line
from vispy.visuals.transforms import STTransform

from ...layers.image.experimental.octree_util import ChunkData
from ...utils.perf import block_timer
from ..image import Image as ImageVisual
from ..vispy_image_layer import VispyImageLayer

# Grid is optionally drawn while debugging to show tile boundaries.
GRID_WIDTH = 3
GRID_COLOR = (1, 0, 0, 1)

# Set order to the grid draws on top of the image tiles.
IMAGE_NODE_ORDER = 0
LINE_VISUAL_ORDER = 10

# We are seeing crashing when creating too many ImageVisual's so we are
# experimenting with having a reusable pool of them.
INITIAL_POOL_SIZE = 30


def _chunk_outline(chunk: ChunkData) -> np.ndarray:
    """Return the line verts that outline the one given chunk.

    Parameters
    ----------
    chunk : ChunkData
        Create outline of this chunk.

    Return
    ------
    np.ndarray
        The chunk verts for a line drawn with the 'segments' mode.
    """
    x, y = chunk.pos
    h, w = chunk.data.shape[:2]
    w *= chunk.scale[1]
    h *= chunk.scale[0]

    # We draw lines on all four sides of the chunk. This means are
    # double-drawing all interior lines in the grid. We can avoid
    # this duplication if it becomes a performance issue.
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
    """Holds the ImageVisual for a single chunk.

    Parameters
    ----------
    chunk : ChunkData
        The data for the ImageChunk.
    node : Optional[ImageVisual]
        The ImageVisual if one was available.
    """

    # TODO_OCTREE: make this a namedtuple if it doesn't grow.
    def __init__(self, chunk_data: ChunkData, node: Optional[ImageVisual]):
        self.chunk_data = chunk_data
        self.data_id = id(chunk_data.data)
        self.node = node  # If None it means no ImageVisual was available


class ImageVisualPool:
    """A reusable pool of ImageVisuals.

    Because we are seeing crashes if we create too many ImageVisuals we have
    a reusable pool of them.

    Attributes
    ----------
    nodes : List[ImageVisual]
        The available nodes in the pool.
    """

    def __init__(self):
        size = INITIAL_POOL_SIZE
        with block_timer("ImageVisualPool.__init__") as event:
            self.nodes = [
                ImageVisual(None, method='auto') for x in range(size)
            ]

        ms = event.duration_ms
        each_ms = event.duration_ms / size
        print(f"ImageVisualPool: Created {size} @ {each_ms}ms each = {ms}ms")

    def get_node(self) -> Optional[ImageVisual]:
        """Get an available ImageVisual.

        Return
        ------
        Optional[ImageVisual]
            An ImageVisual if one was available in the pool.
        """
        if len(self.nodes) == 0:
            # Pool is empty, no visual for now. The ImageChunk might get
            # assigned a ImageVisual later if it stays in view.
            return None

        # Assign an available ImageVisual.
        return self.nodes.pop()

    def return_node(self, node: ImageVisual) -> None:
        """Return a node that's no number being used."""
        if node is not None:
            node.parent = None  # Remove from the Scene Graph.
            self.nodes.append(node)


class TileGrid:
    """The grid that shows the outlines of all the tiles.

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

    def update_grid(self, chunks: List[ImageChunk]) -> None:
        """Update grid for this set of chunks.

        Parameters
        ----------
        chunks : List[ImageChunks]
            Add a grid that outlines these chunks.
        """
        # TODO_OCTREE: create in one go without vstack?
        verts = np.zeros((0, 2), dtype=np.float32)
        for image_chunk in chunks:
            chunk_verts = _chunk_outline(image_chunk.chunk_data)
            verts = np.vstack([verts, chunk_verts])

        self.line.set_data(verts)
        self.verts = verts


class VispyTiledImageLayer(VispyImageLayer):
    """Tiled images using multiple ImageVisuals.

    Render a set of image tiles. For example render the set of tiles that
    need to be drawn from an octree in order to depict the current view.

    We create a parent ImageVisual, which is current empty. Then we create a
    child ImageVisual for every image tile. The set of child ImageVisuals will
    change over time as the user pans/zooms in the octree.

    Future Work
    -----------

    This class will likely need to be replaced at some point. We plan to
    write a TiledImageVisual class which stores the tile textures in an
    atlas-like tile cache. Then in one draw command it can draw the all the
    visible tiles in the right positions using texture coordinates.

    One reason TiledImageVisual will be faster is today calling
    ImageVisual.set_data() causes a 15-20ms hiccup when that visual is next
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
        self.pool = ImageVisualPool()

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
        # If no node is available then get_node() returns None and we
        # create the ImageChunk with no ImageVisual. One can be added
        # later on if it becomes available.
        node = self.pool.get_node()
        image_chunk = ImageChunk(chunk_data, node)

        if node is not None:
            self._add_image_chunk_node(image_chunk)

        return image_chunk

    @property
    def _tiled_visual_parent(self):
        """Return the parent under which ImageVisuals should be added."""
        return self.node  # This is VispyBaseLayer.node

    def _add_image_chunk_node(self, image_chunk: ImageChunk) -> None:
        """Add an ImageChunk's node to the scene.

        Parameters
        ----------
        image_chunk : ImageChunk
            Add this chunk's node to the scene.
        """
        node = image_chunk.node
        assert node

        chunk_data = image_chunk.chunk_data

        # Call VispyImageLayer._set_node_data() to process the data assign
        # it to the ImageVisual node.
        self._set_node_data(node, chunk_data.data)

        # Add the node under us, transformed into the right place.
        node.parent = self._tiled_visual_parent
        node.transform = STTransform(chunk_data.scale, chunk_data.pos)
        node.order = IMAGE_NODE_ORDER

    def _on_data_change(self, event=None) -> None:
        """Our self.layer._data_view has been updated, update our node.
        """
        if not self.layer.loaded or not self.ready:
            return

        self._update_image_chunks()

    def _update_image_chunks(self) -> None:
        """Update all ImageChunks that we are managing.

        The basic algorithm is:
        1) Assign ImageChunks and/or ImageVisuals to any chunks that
        are currently visible. We might run out of visuals from the pool.

        2) Remove no longer visible chunks and return them to the pool.

        3) Create the optional grid around only the visible chunks.
        """
        # Get the currently visible chunks.
        visible_chunks = self.layer.visible_chunks

        num_seen = len(visible_chunks)

        # Data id's of the visible chunks
        visible_ids = set([id(chunk.data) for chunk in visible_chunks])

        num_start = len(self.image_chunks)

        # Create ImageChunks/ImageVisuals for all visible chunks.
        self._update_visible(visible_chunks)

        num_peak = len(self.image_chunks)
        num_created = num_peak - num_start

        # Remove chunks no longer visible.
        self._remove_stale_chunks(visible_ids)

        num_final = len(self.image_chunks)
        num_deleted = num_peak - num_final

        self.grid.update_grid(self.image_chunks.values())

        num_pool = len(self.pool.nodes)

        if num_created > 0 or num_deleted > 0:
            print(
                f"VispyTiled: seen: {num_seen} start: {num_start} created: {num_created} "
                f"deleted: {num_deleted} final: {num_final} pool: {num_pool}"
            )

    def _update_visible(self, visible_chunks: List[ChunkData]) -> None:
        """Create or update all visible ImageChunks.

        Go through all visible chunks:
        1) Create a new ImageChunk if one doesn't exist.
        2) Create a new ImageVisual if the existing ImageChunk does
           not have a visual, because none were available in the pool.

        Parameters
        ----------
        visible_chunks : List[ChunkData]
        """
        for chunk in visible_chunks:
            chunk_id = id(chunk.data)
            try:
                image_chunk = self.image_chunks[chunk_id]
                if image_chunk.node is None:
                    # The ImageChunk already existed but there was no
                    # ImageVisual available. Attempt to assign one from the
                    # pool again. If still not available we do nothing.
                    node = self.pool.get_node()
                    if node is not None:
                        image_chunk.node = node
                        self._add_image_chunk_node(image_chunk)
            except KeyError:
                # There is no ImageChunk for this ChunkData, so create a
                # new ImageChunk. It will get an ImageVisual if one is
                # available from the pool. Otherwise maybe its ImageVisual
                # will be assigned later.
                self.image_chunks[chunk_id] = self._create_image_chunk(chunk)

    def _remove_stale_chunks(self, visible_ids: Set[int]) -> None:
        """Remove stale chunks which are not longer visible.

        Parameters
        ----------
        visible_ids : Set[int]
            The data_id's of the currently visible chunks.
        """
        for image_chunk in list(self.image_chunks.values()):
            data_id = image_chunk.data_id
            if data_id not in visible_ids:
                self.pool.return_node(image_chunk.node)
                del self.image_chunks[data_id]

    def _on_camera_move(self, event=None):
        self._update_image_chunks()
