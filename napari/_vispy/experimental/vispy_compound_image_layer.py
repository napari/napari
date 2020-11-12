"""VispyCompoundImageLayer class.
"""
from collections import namedtuple
from typing import List, Set

from vispy.visuals.transforms import STTransform

from ...layers.image.experimental.octree_util import ChunkData
from ...utils.perf import block_timer
from ..image import Image as ImageVisual
from ..vispy_image_layer import VispyImageLayer
from .tile_grid import TileGrid

# Set order to the grid draws on top of the image tiles.
IMAGE_NODE_ORDER = 0

# We are seeing crashing when creating too many ImageVisual's so we are
# experimenting with having a reusable pool of them.
INITIAL_POOL_SIZE = 0

# TODO_OCTREE: hook up to QtRender UI
SHOW_GRID = True

Stats = namedtuple(
    'Stats', "num_seen num_start num_created num_deleted num_final"
)


class NullImageVisualPool:
    """Allocate visuals with no pool.

    We've experimented with visual pools to work around crashes with PyQt5.
    Current we have no pool: we only work with PySide2.
    """

    @staticmethod
    def get_visual():
        """Just allocated the ImageVisual directly."""
        return ImageVisual(None, method='auto')

    @staticmethod
    def return_visual(visual):
        """Just remove it from the scene graph."""
        assert visual
        visual.parent = None  # Remove from scene graph.


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
    def __init__(self, chunk_data: ChunkData):
        self.chunk_data = chunk_data
        self.data_id = id(chunk_data.data)

        # ImageVisual should be assigned later
        self.node = None


class VispyCompoundImageLayer(VispyImageLayer):
    """Tiled images using multiple ImageVisuals.

    This was our initial rendering approach for the octree work. This layer
    creates and manages tiles where each tile is a separate ImageVisual.
    The ImageVisuals are all children of our VispyImageLayer node. We also
    draw a grid under our VispyImageLayer node for debugging.

    This works and produces good looking results. However creating separate
    ImageVisuals is fairly slow. We think partly because each one rebuilds
    its shader when it is created. Also as of Fall 2020 we were seeing
    crashes that we think are related to modifying the scene graph. The
    crashes were only under PyQt5, not with PySide2.

    For these reasons we are going to create a new TiledImageVisual that
    internally manages tiles using a texture atlas. The advantage is that
    it's one single visual so there are no scene graph operations when
    adding or removing tiles. Also adding and removing tiles should be
    fast, since the shader will not change. Finally rendering should be
    fast, since in one draw operation we will draw all the tiles that
    make up the image.
    """

    def __init__(self, layer):
        self.ready = False
        self.image_chunks = {}
        self.test_chunks = []
        self.grid = None  # Can't create until after super() init

        self.pool = NullImageVisualPool()

        # This will call our self._on_data_change() but we guard
        # that with self.read as a hack.
        super().__init__(layer)

        if SHOW_GRID:
            self.grid = TileGrid(self.node)

        self.ready = True

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
        node.transform = STTransform(chunk_data.scale, chunk_data.pos)
        node.parent = self._tiled_visual_parent
        node.order = IMAGE_NODE_ORDER

    def _on_data_change(self, event=None) -> None:
        """Our self.layer._data_view has been updated, update our node.
        """
        if not self.layer.loaded or not self.ready:
            return

        # self._update_image_chunks()

    def _update_view(self) -> None:
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

        # Set is keyed by id(chunk_data.data) and chunk_data.level_index
        visible_set = set(c.key for c in visible_chunks)

        num_start = len(self.image_chunks)

        # Remove chunks no longer visible.
        self._remove_stale_chunks(visible_set)

        num_low = len(self.image_chunks)
        num_deleted = num_start - num_low

        # Create ImageChunks/ImageVisuals for all visible chunks.
        self._update_visible(visible_chunks)

        num_final = len(self.image_chunks)
        num_created = num_final - num_low

        num_final = len(self.image_chunks)

        if SHOW_GRID:
            chunk_datas = [
                image_chunk.chunk_data
                for image_chunk in self.image_chunks.values()
            ]
            self.grid.update_grid(chunk_datas)

        return Stats(num_seen, num_start, num_created, num_deleted, num_final)

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
        track_view = self.layer.track_view

        for chunk_data in visible_chunks:
            if chunk_data.key not in self.image_chunks:
                # Create an ImageChunk for this ChunkData.
                self.image_chunks[chunk_data.key] = ImageChunk(chunk_data)

            if not track_view:
                # We are not actively create ImageVisuals, for example to
                # "freeze" the view for debugging. So we're done.
                continue

            # Whether we just created this ImageChunk or it's older, if it
            # doesn't have an ImageVisual try to add one one
            image_chunk = self.image_chunks[chunk_data.key]
            if image_chunk.node is None:
                # The ImageChunk already existed but there was no
                # ImageVisual available. Attempt to assign one from the
                # pool again. If still not available we do nothing.
                image_chunk.node = self.pool.get_visual()
                self._add_image_chunk_node(image_chunk)

    def _remove_stale_chunks(self, visible_set: Set[ChunkData]) -> None:
        """Remove stale chunks which are not longer visible.

        Parameters
        ----------
        visible_ids : Set[int]
            The data_id's of the currently visible chunks.
        """
        for image_chunk in list(self.image_chunks.values()):
            chunk_data = image_chunk.chunk_data
            if chunk_data.key not in visible_set:
                if image_chunk.node is not None:
                    self.pool.return_visual(image_chunk.node)
                del self.image_chunks[chunk_data.key]

    def _on_camera_move(self, event=None):
        super()._on_camera_move()

        if not self.node.visible:
            return

        with block_timer("_update_view") as elapsed:
            stats = self._update_view()

        if stats.num_created > 0 or stats.num_deleted > 0:
            print(
                f"tiles: {stats.num_start} -> {stats.num_final} "
                f"create: {stats.num_created} delete: {stats.num_deleted} "
                f"time: {elapsed.duration_ms:.3f}ms"
            )
