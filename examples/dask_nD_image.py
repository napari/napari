"""
Display a dask array
"""

try:
    from dask import array as da
except ImportError:
    raise ImportError("""This example uses a dask array but dask is not
    installed. To install try 'pip install dask'.""")

import numpy as np
from skimage import data
import napari


blobs = da.stack(
    [
        data.binary_blobs(
            length=128, blob_size_fraction=0.05, n_dim=3, volume_fraction=f
        )
        for f in np.linspace(0.05, 0.5, 10)
    ],
    axis=0,
)
viewer = napari.view_image(blobs.astype(float))

napari.run()
