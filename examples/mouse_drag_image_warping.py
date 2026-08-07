"""
Image warping with mouse event callbacks
========================================

Warp an image based on points moved by user using mouse event callbacks in
napari.

Derived from scikit-image example: https://scikit-image.org/docs/stable/auto_examples/transform/plot_tps_deformation.html

This example is fully explained in the following tutorial from SciPy 2025:
https://napari.org/napari-scipy2025-workshop/notebooks/image_warping.html

Credit to Lars Grüter for the original idea and first attempts at implementation at SciPy 2024.

.. tags:: interactivity
"""

from functools import partial

import numpy as np
import skimage as ski

import napari

# Set up base image (to be warped) and points layers
image = ski.data.checkerboard()
src = np.array([[66, 66], [133, 66], [66, 133], [133, 133]])

viewer = napari.Viewer()
checkerboard_image_layer = viewer.add_image(image, name='checkerboard')
source_points_layer = viewer.add_points(
    src, name='source_points', symbol='+', face_color='red', size=5
)
moving_points_layer = viewer.add_points(src.copy(), name='moving_points')

# ensure moving_points layer is in Select mode
moving_points_layer.mode = 'select'


# Define a function to estimate the warping required to transform the destination points
# into the source points, and then apply the warping to the original image data,
# replacing the image layer data in-place.
def warp(
    im_layer: 'napari.layers.Image',
    original_image_data: 'np.ndarray',
    src: 'np.ndarray',
    dst: 'np.ndarray',
) -> None:
    # Warp image using thin-plate spline transformation from skimage.
    tps = ski.transform.ThinPlateSplineTransform()
    tps.estimate(dst, src)
    warped = ski.transform.warp(original_image_data, tps)
    # warped will be in 0..1 floats, so we need to
    # multiply by 255 to get it back to the same range
    # as the original data
    im_layer.data = (warped * 255).astype(original_image_data.dtype)


# Use partial to specify arguments for the warp function because the event
# callback only has access to the moving `points_layer` and the `event` object itself
warp_checkerboard = partial(
    warp,  # the warping function
    checkerboard_image_layer,  # im_layer argument
    image,  # original_image_data argument
    src,  # src argument
)


def warp_on_move(points_layer, event):
    # we do nothing here, as we don't care about the mouse press,
    # by yielding we have the chance to do things on drag
    yield

    # while the mouse is moving, we call our warp function
    while event.type == 'mouse_move':
        # ensure a point is selected and we're in select mode
        if not points_layer.selected_data or points_layer.mode != 'select':
            return
        # find the index into the points data of the currently selected point
        # we use the last selected point as that's likely what the mouse is hovering
        # over
        moved_point_index = list(points_layer.selected_data)[-1]
        # make a copy of the moving_points so original array is unchanged
        dst = points_layer.data.copy()
        # assign the current mouse position into the correct index to
        # update the location of the point
        dst[moved_point_index] = event.position
        # warp image
        warp_checkerboard(dst)
        yield

    # Empty here as nothing is done on mouse release


# Hook up callback to the layer event
moving_points_layer.mouse_drag_callbacks.append(warp_on_move)

if __name__ == '__main__':
    napari.run()
