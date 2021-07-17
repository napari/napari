from vispy.scene.visuals import create_visual_node

from .vendored import VolumeVisual as BaseVolumeVisual

Volume = create_visual_node(BaseVolumeVisual)
