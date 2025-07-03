"""
Launch napari and display an image
==================================

Launch an instance of napari and simultaneously display an image using the
:func:`imshow` API. This example can be run to recreate the README image.

.. tags:: visualization-basic
"""

from skimage import data

import napari

viewer, layers = napari.imshow(data.cells3d(), channel_axis=1, ndisplay=3)

# To recreate the README image uncomment the following lines:
# viewer.window._qt_window.resize(800, 600)
# napari._qt.qt_event_loop.get_qapp().processEvents()  # Ensure the window is done rendering before taking a screenshot
# viewer.screenshot('./src/napari/resources/multichannel_cells.png', canvas_only=False)

if __name__ == '__main__':
    napari.run()
