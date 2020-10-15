from .octree_image import OctreeImage


class ChunkData:
    def __init__(self, x, y, data):
        self.x = x
        self.y = y
        self.data = data
