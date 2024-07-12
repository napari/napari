"""
Labels 3D
=========

View 3D labels.

.. tags:: visualization-nD
"""


import magicgui
import numpy as np
from scipy import ndimage as ndi
from skimage import data, filters, morphology

import napari

cells3d = data.cells3d()

viewer = napari.view_image(
    cells3d, channel_axis=1, name=['membranes', 'nuclei']
)
membrane, nuclei = cells3d.transpose((1, 0, 2, 3)) / np.max(cells3d)
edges = filters.scharr(nuclei)
denoised = ndi.median_filter(nuclei, size=3)
thresholded = denoised > filters.threshold_li(denoised)
cleaned = morphology.remove_small_objects(
    morphology.remove_small_holes(thresholded, 20**3),
    20**3,
)

segmented = ndi.label(cleaned)[0]
# maxima = ndi.label(morphology.local_maxima(filters.gaussian(nuclei, sigma=10)))[0]
# markers_big = morphology.dilation(maxima, morphology.ball(5))

# segmented = segmentation.watershed(
#     edges,
#     markers_big,
#     mask=cleaned,
# )

labels_layer = viewer.add_labels(segmented)


@magicgui.magicgui
def toggle_smooth_labels(viewer: napari.viewer.Viewer, layer: napari.layers.Labels):
    if viewer.dims.ndisplay != 3:
        return
    node = viewer.window.qt_viewer.layer_to_visual[layer].node
    node.iso_gradient = not node.iso_gradient


viewer.window.add_dock_widget(toggle_smooth_labels)
viewer.dims.ndisplay = 3

if __name__ == '__main__':
    napari.run()
