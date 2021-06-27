"""
Display a 3D volume and the scale bar
"""
import numpy as np
import napari
from skimage import data

cells = data.cells3d()

viewer = napari.Viewer(ndisplay=3)

viewer.add_image(
    cells,
    name=('membrane', 'nuclei'),
    channel_axis=1,
    scale=(0.29, 0.26, 0.26),
)
viewer.scale_bar.visible = True
viewer.scale_bar.unit = "um"

napari.run()
