"""Progressive loading of a whole slide image (WSI) from S3.

Uses the public CMU-1 Aperio SVS dataset from Glencoe Software's
S3 bucket, converted to OME-Zarr via bioformats2raw.

Run:  .venv/bin/python examples/progressive_loading_wsi_.py
"""

import dask.array as da
import zarr
from zarr.storage import FsspecStore

import napari
from napari.experimental._progressive_loading import (
    add_progressive_loading_image,
)

BUCKET = 's3://gs-public-zarr-archive/CMU-1.ome.zarr'
NUM_LEVELS = 5

store = FsspecStore.from_url(BUCKET, storage_options={'anon': True})
group = zarr.open_group(store, mode='r', zarr_format=2)
series = group['0']

arrays = []
for i in range(NUM_LEVELS):
    arr = da.from_zarr(series[str(i)]).squeeze()  # (3, H, W)
    rgb = arr.transpose(1, 2, 0).rechunk({2: 3})  # (H, W, 3)
    arrays.append(rgb)
    print(f"level {i}: {rgb.shape}  chunks={rgb.chunksize}")

viewer = napari.Viewer()
layer = add_progressive_loading_image(
    arrays,
    viewer=viewer,
    contrast_limits=(0, 255),
    colormap='gray',
    name='CMU-1 (WSI)',
)

if __name__ == '__main__':
    napari.run()
