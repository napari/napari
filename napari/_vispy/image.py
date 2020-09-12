from vispy.scene.visuals import create_visual_node

from .vendored import ComplexImageVisual, ImageVisual

Image = create_visual_node(ImageVisual)
ComplexImage = create_visual_node(ComplexImageVisual)
