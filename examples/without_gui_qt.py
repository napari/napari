"""
Alternative to using napari.gui_qt() context manager. 
"""

from skimage import data
import napari

viewer = napari.view_image(data.astronaut(), rgb=True)
# You can also take screenshots before running the app
screenshot = viewer.screenshot()

print('Maximum value', screenshot.max())

# Run the app to see the napari viewer. If you only wanted the screenshot
# then you could skip this entirely
napari.run_app()
