"""
Display a 3D volume and the scale bar
"""
import numpy as np
import napari


viewer = napari.Viewer()
viewer.add_image(np.random.random((5, 5, 5)), colormap='red', opacity=0.8)
viewer.scale_bar.visible = True

napari.run()
