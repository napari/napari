"""VispyTiledImageLayer class.
"""
import numpy as np
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

    def _outline_chunk(self, data):
        line = np.array([255, 0, 0])
        data[0, :, :] = line
        data[-1, :, :] = line
        data[:, 0, :] = line
        data[:, -1, :] = line
        return data

    def _create_image_chunk(self, chunk: ChunkData):
        """Add a new chunk.

        Parameters
        ----------
        chunk : ChunkData
            The data used to create the new chunk.
        """
        image_chunk = ImageChunk()

        data = self._outline_chunk(chunk.data)

        # Parent VispyImageLayer will process the data then set it.
        self._set_node_data(image_chunk.node, data)

        # Make the new ImageChunk a child positioned with us.
        image_chunk.node.parent = self.node
        pos = [chunk.pos[0] * 1024, chunk.pos[1] * 1024]
        size = chunk.size * 16
        # pos = [512, 0]
        # size = 7

        # print(pos, size)

        image_chunk.node.transform = STTransform(
            translate=pos, scale=[size, size]
        )

        return image_chunk

    def _on_data_change(self, event=None) -> None:
        """Our self.layer._data_view has been updated, update our node.
        """
        if not self.layer.loaded:
            # Do nothing if we are not yet loaded.
            return

        for chunk in self.layer.view_chunks:
            chunk_id = id(chunk.data)
            if chunk_id not in self.chunks:
                # print(f"Adding chunk {chunk_id}")
                self.chunks[chunk_id] = self._create_image_chunk(chunk)
