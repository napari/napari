from .vendored import ImageVisual
from vispy.scene.visuals import create_visual_node


Image = create_visual_node(ImageVisual)
