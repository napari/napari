"""VispyTiledImageLayer class.
"""
import numpy as np
from vispy.scene.visuals import Compound, Line
from vispy.visuals.transforms import STTransform

from ...layers.image.experimental.octree import ChunkData
from ..image import Image as ImageNode
from ..vispy_image_layer import VispyImageLayer


class ImageChunk:
    def __init__(self):
        self.node = ImageNode(None, method='auto')


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

        # This will call our self._on_data_change().
        super().__init__(layer)

        self.line = Line(connect='segments', color=(1, 0, 0, 1), width=10)
        self.compound = Compound([self.line])
        self.compound.parent = self.node

        self.line_verts = np.zeros((0, 2), dtype=np.float32)

    def _create_image_chunk(self, chunk: ChunkData):
        """Add a new chunk.

        Parameters
        ----------
        chunk : ChunkData
            The data used to create the new chunk.
        """
        image_chunk = ImageChunk()

        x, y = chunk.pos
        h, w = chunk.data.shape[:2]
        w *= chunk.scale[1]
        h *= chunk.scale[0]

        pos = np.zeros((8, 2), dtype=np.float32)
        pos[0, :] = [x, y]
        pos[1, :] = [x + w, y]

        pos[2, :] = [x + w, y]
        pos[3, :] = [x + w, y + h]

        pos[4, :] = [x + w, y + h]
        pos[5, :] = [x, y + h]

        pos[6, :] = [x, y + h]
        pos[7, :] = [x, y]
        self.line_verts = np.vstack([self.line_verts, pos])
        self.line.set_data(self.line_verts)

        # data = self._outline_chunk(chunk.data)

        # Parent VispyImageLayer will process the data then set it.
        self._set_node_data(image_chunk.node, chunk.data)

        # Add this new ImageChunk as child of self.node, transformed into place.
        # image_chunk.node.parent = self.node
        image_chunk.node.transform = STTransform(chunk.scale, chunk.pos)

        return image_chunk

    def _on_data_change(self, event=None) -> None:
        """Our self.layer._data_view has been updated, update our node.
        """
        if not self.layer.loaded:
            # Do nothing if we are not yet loaded.
            return

        # For now, nuke all the old chunks.
        for image_chunk in self.chunks.values():
            image_chunk.node.parent = None
        self.chunks = {}
        self.line_verts = np.zeros((0, 2), dtype=np.float32)

        for chunk in self.layer.view_chunks:
            chunk_id = id(chunk.data)
            if chunk_id not in self.chunks:
                self.chunks[chunk_id] = self._create_image_chunk(chunk)
