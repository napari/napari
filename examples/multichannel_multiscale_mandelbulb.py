import dask.array as da
import numpy as np
import zarr

from napari.experimental._generative_zarr import MultichannelMandelbulbStore
from napari.experimental._progressive_loading import (
    add_progressive_loading_image,
)

def mandelbulb_dataset(max_levels=14, channels=[4, 5,6 ]):
    """Generate a multiscale image of the mandelbulb set for a given number
    of levels/scales. Scale 0 will be the highest resolution.

    ...
    """
    chunk_size = (1, 32, 32, 32)

    # Initialize the store with MultichannelMandelbulbStore
    store = zarr.storage.KVStore(
        MultichannelMandelbulbStore(
            levels=max_levels,
            tilesize=32,
            channels=channels,
            compressor=None,
            maxiter=255
        )
    )
    # Wrap in a cache so that tiles don't need to be computed as often
    # store = zarr.LRUStoreCache(store, max_size=8e9)

    # This store implements the 'multiscales' zarr specfiication which is recognized by vizarr
    z_grp = zarr.open(store, mode="r")

    multiscale_img = [z_grp[str(k)] for k in range(max_levels)]

    arrays = []
    for _scale, a in enumerate(multiscale_img):
        da.core.normalize_chunks(
            chunk_size,
            a.shape,
            dtype=np.uint8,
            previous_chunks=None,
        )

        arrays += [a]

    return arrays


if __name__ == "__main__":
    import napari

    viewer = napari.Viewer(ndisplay=3)


    multiscale_img = mandelbulb_dataset(max_levels=16)

    print(multiscale_img[0])

    add_progressive_loading_image(
        multiscale_img,
        viewer=viewer,
        contrast_limits=[0, 255],
        colormap='twilight_shifted',
        ndisplay=3,
        scale=(1, 1, 1, 1),
    )

    viewer.axes.visible = True

# viewer.layers[0].metadata["should_render"] = False
