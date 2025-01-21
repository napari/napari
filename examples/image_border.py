"""
Border Overlay / Bounding Box
=============================

Display a 3D, 2 layer image in napari and add a border overlay (i.e. `bounding_box`).
Creates two viewers to show functionality in both 2D and 3D display mode.

The 2D viewer is utilized to demonstrate visualization of many border overlay properties,
including line color, line thickness, point size, opacity.
In addition, this viewer shows how layer overlays interact when draw on top of each other.
Use `viewer.layers[0].bounding_box` to view all modifiable attributes of the border overlay.

The 3D viewer shows default `bounding_box` properties (changing only color), use in grid mode, and extent of 3D bounding box.

.. tags:: visualization-advanced, visualization-nD
"""

from skimage import data

import napari

# Create a napari viewer with a 3D, 2 channel image, with the `view_image` convenience function.
viewer_2d = napari.view_image(
    data.cells3d(), channel_axis=1, name=['membrane', 'nuclei']
)
viewer_2d.grid.enabled = True

# Add a border overlay to each layer, and modify properties of the border overlay.
viewer_2d.layers[0].bounding_box.visible = True
viewer_2d.layers[0].bounding_box.line_color = 'cyan' # default: 'red'
viewer_2d.layers[0].bounding_box.line_thickness = 5 # default: 1, max: 5
viewer_2d.layers[0].bounding_box.point_size = 10 # default: 5
viewer_2d.layers[0].bounding_box.point_color = 'yellow' # default: 'blue'

viewer_2d.layers[1].bounding_box.visible = True
viewer_2d.layers[1].bounding_box.line_color = 'orange'
viewer_2d.layers[1].bounding_box.line_thickness = 2
viewer_2d.layers[1].bounding_box.opacity = 0.8 # default: 1
# viewer_2d.layers[1].bounding_box.blending = 'additive' # default: 'translucent'


# An equivalent alternative to creating the 2D napari viewer, this viewer is
# created with 3D display and the `open_sample` convenience function
viewer_3d = napari.view_image(
    data.cells3d(), channel_axis=1, name=['membrane', 'nuclei']
)
viewer_3d.grid.enabled = True
viewer_3d.dims.ndisplay = 3
viewer_3d.camera.angles = (5, 25, 130)

viewer_3d.layers[0].bounding_box.visible = True
viewer_3d.layers[1].bounding_box.visible = True
viewer_3d.layers[1].bounding_box.line_color = 'cyan'

if __name__ == '__main__':
    napari.run()
