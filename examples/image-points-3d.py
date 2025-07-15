"""
Image points 3D
===============

Display points overlaid on a 3D image

.. tags:: visualization-nD
"""
from skimage import data, feature, filters

import napari

cells = data.cells3d()
nuclei = cells[:, 1]
smooth = filters.gaussian(nuclei, sigma=10)
pts = feature.peak_local_max(smooth)
viewer = napari.Viewer(ndisplay=3)
membranes, nuclei = viewer.add_image(
    cells, channel_axis=1, name=['membranes', 'nuclei']
)
viewer.add_points(pts)
viewer.camera.angles = (10, -20, 130)

if __name__ == '__main__':
    napari.run()
