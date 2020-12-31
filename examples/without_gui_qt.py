"""
Alternative to using napari.gui_qt() context manager. 
"""

from skimage import data
import napari

viewer = napari.view_image(data.astronaut(), rgb=True)

napari.run_app()
