"""VispyTiledImageLayer class.
"""
from .vispy_image_layer2 import VispyImageLayer2


class VispyTiledImageLayer(VispyImageLayer2):
    """Tiled images using multiple ImageNodes.

    Render a set of image tiles. For example render the set of tiles that
    need to be drawn from an octree in order to depict the current view.

    We create a parent ImageNode, which is current empty. Then we create a
    child ImageNode for every image tile. The set of child ImageNodes will
    change over time as the user pans/zooms in the octree.

    This class will likely need to be replaced at some point. We plan to
    write a TiledImageVisual class which stores the tile textures in an
    atlas-like tile cache. Then in one draw command it can draw the
    currently visible tiles in the right positions, using the right texture
    coordinates.

    One reason TiledImageVisual will be faster is today calling
    ImageNode.set_data() causes a 15-20ms hiccup when that visual is next
    drawn. It's not clear why the hiccup is so big, since transfering the
    texture into VRAM should be under 2ms. However at least 4ms is spent
    rebuilding the shader alone.

    Since we might need to draw 10+ new tiles at one time, the delay could
    be up to 200ms total, which might be quite a noticeable glitch.

    However, this initial approach with separate ImageVisuals should be a
    good starting point, since it will let us iron out the octree
    implementation and the tile placement math. Then we can upgrade
    to the new TiledImageVisual when its ready, and everything should
    still work fine.
    """

    def __init__(self, layer):
        super().__init__(layer)
