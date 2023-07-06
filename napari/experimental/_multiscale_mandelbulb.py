import dask.array as da
import numpy as np
import zarr

from napari.experimental._progressive_loading import (
    add_progressive_loading_image,
)

from napari.experimental._generative_zarr import MandelbulbStore


# https://dask.discourse.group/t/using-da-delayed-for-zarr-processing-memory-overhead-how-to-do-it-better/1007/10
def mandelbulb_dataset(max_levels=14):
    """Generate a multiscale image of the mandelbulb set for a given number
    of levels/scales. Scale 0 will be the highest resolution.

    This is intended to be used with progressive loading. As such, it returns
    a dictionary will all the metadata required to load as multiple scaled
    image layers via add_progressive_loading_image

    >>> large_image = mandelbrot_dataset(max_levels=14)
    >>> multiscale_img = large_image["arrays"]
    >>> viewer._layer_slicer._force_sync = False
    >>> add_progressive_loading_image(multiscale_img, viewer=viewer)

    Parameters
    ----------
    max_levels: int
        Maximum number of levels (scales) to generate

    Returns
    -------
    Dictionary of multiscale data with keys ['container', 'dataset',
        'scale levels', 'scale_factors', 'chunk_size', 'arrays']
    """

    large_image = {
        "container": "mandelbulb.zarr/",
        "dataset": "",
        "scale_levels": max_levels,
        "scale_factors": [
            (2**level, 2**level, 2**level) for level in range(max_levels)
        ],
        "chunk_size": (32, 32, 32),
    }

    # Initialize the store
    store = zarr.storage.KVStore(
        MandelbulbStore(
            levels=max_levels,
            tilesize=32,
            compressor=None,
            maxiter=255
            #        levels=max_levels, tilesize=512, compressor=Blosc(), maxiter=255
        )
    )
    # Wrap in a cache so that tiles don't need to be computed as often
    # store = zarr.LRUStoreCache(store, max_size=8e9)

    # This store implements the 'multiscales' zarr specfiication which is recognized by vizarr
    z_grp = zarr.open(store, mode="r")

    multiscale_img = [z_grp[str(k)] for k in range(max_levels)]

    arrays = []
    for scale, a in enumerate(multiscale_img):
        chunks = da.core.normalize_chunks(
            large_image["chunk_size"],
            a.shape,
            dtype=np.uint8,
            previous_chunks=None,
        )

        # arrays += [da.from_zarr(a, chunks=chunks)]

        setattr(
            a,
            "get_zarr_chunk",
            lambda scale, z, y, x: store.get_chunk(scale, z, y, x),
        )
        # setattr(a, "get_zarr_chunk", lambda chunk_slice: a[tuple(chunk_slice)].transpose())
        arrays += [a]

    large_image["arrays"] = arrays

    # TODO wrap in dask delayed

    return large_image


if __name__ == "__main__":
    import napari

    viewer = napari.Viewer(ndisplay=3)

    large_image = mandelbulb_dataset(max_levels=15)

    multiscale_img = large_image["arrays"]

    print(multiscale_img[0])

    add_progressive_loading_image(
        multiscale_img,
        viewer=viewer,
        contrast_limits=[0, 255],
        colormap='twilight_shifted',
        ndisplay=3,
    )

    # layer = viewer.add_image(multiscale_img[0], name="volume")
    # layer.bounding_box.visible = True

    viewer.axes.visible = True

    # viewer.text_overlay.update(dict(
    #     text='Click and drag the clipping plane surface to move it along its normal.',
    #     visible=True,
    # ))
