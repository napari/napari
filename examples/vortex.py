"""Visualizing optical flow in napari.

Adapted from the scikit-image gallery [1]_.

In napari, we can show the flowing vortex as an additional dimension in the
image, visible by moving the slider. We can also take advantage of the reverse
Phi illusion [2]_ to simulate a vortex continuously flowing in one direction.
To do this, we repeat the two vortex frames with inverted contrast at the end.
When using the slider Play button, we perceive a continously flowing vortex
aligned with the displayed vector field.

.. tags:: visualization-advanced, layers

.. [1] https://scikit-image.org/docs/stable/auto_examples/registration/plot_opticalflow.html

.. [2] Anstis, S. M. & Rogers, B. J. (1986) Illusory continuous motion from
       oscillating positive-negative patterns: implications for motion
       perception. Perception, 15, 627-640. :DOI:`10.1068/p150627`
"""
import numpy as np
from skimage.data import vortex
from skimage.registration import optical_flow_ilk

import napari

#######################################################################
# First, we load the vortex image, and then we create an inverted
# "negative" image. For the illusion, we concatenate them end-to-end.

vortex_im = np.asarray(vortex())
vortex_neg = np.max(vortex_im) - vortex_im
vortex_phi = np.concatenate((vortex_im, vortex_neg), axis=0)

#######################################################################
# We compute the optical flow using scikit-image. (Note: as of
# scikit-image 0.21, there seems to be a transposition of the image in
# the output, which we account for later.)

u, v = optical_flow_ilk(vortex_im[0], vortex_im[1], radius=15)

#######################################################################
# Compute the flow magnitude, for visualization.

magnitude = np.sqrt(u ** 2 + v ** 2)

#######################################################################
# Create a viewer, add the vortex frames, and overlay the flow
# magnitude.

viewer, vortex_layer = napari.imshow(vortex_phi)
mag_layer = viewer.add_image(magnitude, colormap='magma', opacity=0.3)

#######################################################################
# Finally, we subsample the vector field to display it — it's too
# messy otherwise! And we transpose the rows/columns axes to match the
# current scikit-image output.

nvec = 21
nr, nc = magnitude.shape
step = max(nr//nvec, nc//nvec)
offset = step // 2
usub = u[offset::step, offset::step]
vsub = v[offset::step, offset::step]

vectors_field = np.transpose(  # transpose required — skimage bug?
        np.stack([usub, vsub], axis=-1),
        (1, 0, 2),
        )

flow_layer = viewer.add_vectors(
        vectors_field,
        scale=[step, step],
        translate=[offset, offset],
        edge_width=0.3,
        length=0.3,
        )

if __name__ == '__main__':
    napari.run()
