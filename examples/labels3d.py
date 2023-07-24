"""
Labels 3D
=========

View 3D labels.

.. tags:: visualization-nD
"""


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

if __name__ == '__main__':
    napari.run()
