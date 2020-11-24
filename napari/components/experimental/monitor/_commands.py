"""MonitorCommands class.
"""

from ....layers.image.experimental.octree_image import OctreeImage
from ...layerlist import LayerList


class MonitorCommands:
    def __init__(self, layers: LayerList):
        self.layers = layers

    def show_grid(self, value: bool) -> None:
        print(f"SHOW GRID: {value}")
        for layer in self.layers.selected:
            if isinstance(layer, OctreeImage):
                layer.show_grid = value
