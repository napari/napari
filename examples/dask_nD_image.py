"""
Dask nD image
=============

Display a dask array

.. tags:: visualization-nD
"""

try:
    from dask import array as da
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        """This example uses a dask array but dask is not
    installed. To install try 'pip install dask'."""
    )

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

if __name__ == '__main__':
    napari.run()
