import logging
import sys

import dask.array as da
import numpy as np
import zarr
from fibsem_tools import read_xarray
from numcodecs import Blosc
from ome_zarr.io import parse_url

from napari.experimental._generative_zarr import MandelbrotStore

LOGGER = logging.getLogger("napari.experimental._progressive_loading_datasets")
LOGGER.setLevel(logging.DEBUG)

streamHandler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
streamHandler.setFormatter(formatter)
LOGGER.addHandler(streamHandler)


# TODO capture some sort of metadata about scale factors
def openorganelle_mouse_kidney_labels():
    large_image = {
        "container": "s3://janelia-cosem-datasets/jrc_mus-kidney/jrc_mus-kidney.n5",
        "dataset": "labels/empanada-mito_seg",
        "scale_levels": 4,
        "scale_factors": [(1, 1, 1), (2, 2, 2), (4, 4, 4), (8, 8, 8)],
    }
    large_image["arrays"] = [
        read_xarray(
            f"{large_image['container']}/{large_image['dataset']}/s{scale}/",
            storage_options={"anon": True},
        ).data
        for scale in range(large_image["scale_levels"])
    ]
    return large_image


def openorganelle_mouse_kidney_em():
    large_image = {
        "container": "s3://janelia-cosem-datasets/jrc_mus-kidney/jrc_mus-kidney.n5",
        "dataset": "em/fibsem-uint8",
        "scale_levels": 5,
        "scale_factors": [
            (1, 1, 1),
            (2, 2, 2),
            (4, 4, 4),
            (8, 8, 8),
            (16, 16, 16),
        ],
    }
    large_image["arrays"] = [
        read_xarray(
            f"{large_image['container']}/{large_image['dataset']}/s{scale}/",
            storage_options={"anon": True},
        ).data
        for scale in range(large_image["scale_levels"])
    ]
    return large_image


# TODO this one needs testing, it is chunked over 5D
def idr0044A():
    large_image = {
        "container": "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.4/idr0044A/4007801.zarr",
        "dataset": "",
        "scale_levels": 5,
        "scale_factors": [
            (1, 1, 1),
            (1, 2, 2),
            (1, 4, 4),
            (1, 8, 8),
            (1, 16, 16),
        ],
    }
    large_image["arrays"] = [
        read_xarray(
            f"{large_image['container']}/{scale}/",
            #            storage_options={"anon": True},
        ).data.rechunk((1, 1, 128, 128, 128))
        # .data[362, 0, :, :, :].rechunk((512, 512, 512))
        for scale in range(large_image["scale_levels"])
    ]
    return large_image


def idr0075A():
    large_image = {
        "container": "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.3/idr0075A/9528933.zarr",
        "dataset": "",
        "scale_levels": 4,
        "scale_factors": [(1, 1, 1), (1, 2, 2), (1, 4, 4), (1, 8, 8)],
    }
    large_image["arrays"] = [
        read_xarray(
            f"{large_image['container']}/{scale}/",
            #            storage_options={"anon": True},
        ).data
        # .data[362, 0, :, :, :].rechunk((512, 512, 512))
        for scale in range(large_image["scale_levels"])
    ]
    # .rechunk((1, 1, 128, 128, 128))
    return large_image


def idr0051A():
    large_image = {
        "container": "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.3/idr0051A/4007817.zarr",
        "dataset": "",
        "scale_levels": 3,
        "scale_factors": [(1, 1, 1), (1, 2, 2), (1, 4, 4)],
    }
    large_image["arrays"] = [
        read_xarray(
            f"{large_image['container']}/{scale}/",
            #            storage_options={"anon": True},
        ).data
        # .data[362, 0, :, :, :].rechunk((512, 512, 512))
        for scale in range(large_image["scale_levels"])
    ]
    # .rechunk((1, 1, 128, 128, 128))
    return large_image


def luethi_zenodo_7144919():
    import os

    import pooch

    # Downloaded from https://zenodo.org/record/7144919#.Y-OvqhPMI0R
    # TODO use pooch to fetch from zenodo
    # zip_path = pooch.retrieve(
    #     url="https://zenodo.org/record/7144919#.Y-OvqhPMI0R",
    #     known_hash=None,# Update hash
    # )
    dest_dir = pooch.retrieve(
        url="https://zenodo.org/record/7144919/files/20200812-CardiomyocyteDifferentiation14-Cycle1.zarr.zip?download=1",
        known_hash="e6773fc97dcf3689e2f42e6504e0d4f4d0845c329dfbdfe92f61c2f3f1a4d55d",
        processor=pooch.Unzip(),
    )
    local_container = os.path.split(dest_dir[0])[0]
    print(local_container)

    store = parse_url(local_container, mode="r").store
    store = zarr.LRUStoreCache(store, max_size=8e9)
    z_grp = zarr.open(store, mode="r")

    large_image = {
        "container": local_container,
        "dataset": "B/03/0",
        "scale_levels": 5,
        "scale_factors": [
            (1, 0.1625, 0.1625),
            (1, 0.325, 0.325),
            (1, 0.65, 0.65),
            (1, 1.3, 1.3),
            (1, 2.6, 2.6),
        ],
        "chunk_size": (1, 10, 256, 256),
    }

    multiscale_data = z_grp[large_image["dataset"]]

    large_image["arrays"] = [
        multiscale_data[str(scale)]
        for scale in range(large_image["scale_levels"])
    ]

    return large_image


# ----- zarr extension -----


def zarr_get_chunk(self: "zarr.Array", coords):
    """Accept a tuple of integers as coordinates.
    Return a numpy array with the corresponding loaded chunk data."""
    out = np.zeros(self.chunks)
    selection = [slice(0, mx, 1) for mx in self._chunks]
    self._chunk_getitems([coords], tuple(selection), out, tuple(selection))
    return out


zarr.Array.get_chunk = zarr_get_chunk


# ----- mandelbrot -----
# derived from the mandelbrot example from vizarr: https://colab.research.google.com/github/hms-dbmi/vizarr/blob/main/example/mandelbrot.ipynb


# def create_meta_store(levels, tilesize, compressor, dtype):
#     store = dict()
#     init_group(store)

#     datasets = [{"path": str(i)} for i in range(levels)]
#     root_attrs = {"multiscales": [{"datasets": datasets, "version": "0.1"}]}
#     store[".zattrs"] = json_dumps(root_attrs)

#     base_width = tilesize * 2**levels
#     for level in range(levels):
#         width = int(base_width / 2**level)
#         init_array(
#             store,
#             path=str(level),
#             shape=(width, width),
#             chunks=(tilesize, tilesize),
#             dtype=dtype,
#             compressor=compressor,
#         )
#     return store


# @njit(nogil=True)
# def xcoord_image(out, from_x, from_y, to_x, to_y, grid_size, maxiter):
#     step_x = (to_x - from_x) / grid_size
#     step_y = (to_y - from_y) / grid_size
#     creal = from_x
#     cimag = from_y
#     for i in range(grid_size):
#         cimag = from_y
#         for j in range(grid_size):
#             out[j * grid_size + i] = i
#             cimag += step_y
#         creal += step_x
#     return out


# @njit()
# def tile_bounds(level, x, y, max_level, min_coord=-2.5, max_coord=2.5):
#     max_width = max_coord - min_coord
#     tile_width = max_width / 2 ** (max_level - level)
#     from_x = min_coord + x * tile_width
#     to_x = min_coord + (x + 1) * tile_width

#     from_y = min_coord + y * tile_width
#     to_y = min_coord + (y + 1) * tile_width

#     return from_x, from_y, to_x, to_y


# https://dask.discourse.group/t/using-da-delayed-for-zarr-processing-memory-overhead-how-to-do-it-better/1007/10
def mandelbrot_dataset(max_levels=14):
    """Generate a multiscale image of the mandelbrot set for a given number
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
        "container": "mandelbrot.zarr/",
        "dataset": "",
        "scale_levels": max_levels,
        "scale_factors": [
            (2**level, 2**level) for level in range(max_levels)
        ],
        "chunk_size": (256, 256),
    }

    # Initialize the store
    store = zarr.storage.KVStore(
        MandelbrotStore(
            levels=max_levels,
            tilesize=512,
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
        da.core.normalize_chunks(
            large_image["chunk_size"],
            a.shape,
            dtype=np.uint8,
            previous_chunks=None,
        )

        # arrays += [da.from_zarr(a, chunks=chunks)]

        a.get_zarr_chunk = lambda scale, y, x: store.get_chunk(scale, y, x)
        # setattr(a, "get_zarr_chunk", lambda chunk_slice: a[tuple(chunk_slice)].transpose())
        arrays += [a]

    large_image["arrays"] = arrays

    # TODO wrap in dask delayed

    return large_image


if __name__ == "__main__":
    max_levels = 16

    luethi_zenodo_7144919()

    large_image = {
        "container": "mandelbrot.zarr/",
        "dataset": "",
        "scale_levels": max_levels,
        "scale_factors": [
            (2**level, 2**level) for level in range(max_levels)
        ],
        "chunk_size": (512, 512),
    }

    # Initialize the store
    store = zarr.storage.KVStore(
        MandelbrotStore(
            levels=max_levels, tilesize=512, compressor=Blosc(), maxiter=255
        )
    )
    # Wrap in a cache so that tiles don't need to be computed as often
    store = zarr.LRUStoreCache(store, max_size=8e9)

    # This store implements the 'multiscales' zarr specfiication which is recognized by vizarr
    z_grp = zarr.open(store, mode="r")

    multiscale_img = [z_grp[str(k)] for k in range(max_levels)]

    arrays = []
    for a in multiscale_img:
        chunks = da.core.normalize_chunks(
            large_image["chunk_size"],
            a.shape,
            dtype=np.uint8,
            previous_chunks=None,
        )

        arrays += [a]
