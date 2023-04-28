import dask.array as da
import napari
import numpy as np
import zarr
from numba import njit
from numcodecs import Blosc
from zarr.storage import init_array, init_group
from zarr.util import json_dumps

from fibsem_tools import read_xarray
from ome_zarr.io import parse_url
from ome_zarr.reader import Reader


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
    reader = Reader(parse_url(local_container))
    nodes = list(reader())
    image_node = nodes[0]
    dask_data = image_node.data

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
    large_image["arrays"] = []
    for scale in range(large_image["scale_levels"]):
        array = dask_data[scale]

        # TODO extract scale_factors now

        # large_image["arrays"].append(result.data.rechunk((3, 10, 256, 256)))
        large_image["arrays"].append(
            array.rechunk((1, 10, 256, 256)).squeeze()
            # result.data[2, :, :, :].rechunk((10, 256, 256)).squeeze()
        )
    return large_image

# ----- mandelbrot -----
# derived from the mandelbrot example from vizarr: https://colab.research.google.com/github/hms-dbmi/vizarr/blob/main/example/mandelbrot.ipynb

def create_meta_store(levels, tilesize, compressor, dtype):
    store = dict()
    init_group(store)

    datasets = [{"path": str(i)} for i in range(levels)]
    root_attrs = {"multiscales": [{"datasets": datasets, "version": "0.1"}]}
    store[".zattrs"] = json_dumps(root_attrs)

    base_width = tilesize * 2**levels
    for level in range(levels):
        width = int(base_width / 2**level)
        init_array(
            store,
            path=str(level),
            shape=(width, width),
            chunks=(tilesize, tilesize),
            dtype=dtype,
            compressor=compressor,
        )
    return store


@njit
def mandelbrot(out, from_x, from_y, to_x, to_y, grid_size, maxiter):
    step_x = (to_x - from_x) / grid_size
    step_y = (to_y - from_y) / grid_size
    creal = from_x
    cimag = from_y
    for i in range(grid_size):
        cimag = from_y
        for j in range(grid_size):
            nreal = real = imag = n = 0
            for _ in range(maxiter):
                nreal = real * real - imag * imag + creal
                imag = 2 * real * imag + cimag
                real = nreal
                if real * real + imag * imag > 4.0:
                    break
                n += 1
            out[j * grid_size + i] = n
            cimag += step_y
        creal += step_x
    return out


@njit
def xcoord_image(out, from_x, from_y, to_x, to_y, grid_size, maxiter):
    step_x = (to_x - from_x) / grid_size
    step_y = (to_y - from_y) / grid_size
    creal = from_x
    cimag = from_y
    for i in range(grid_size):
        cimag = from_y
        for j in range(grid_size):
            nreal = real = imag = n = 0
            for _ in range(maxiter):
                nreal = real * real - imag * imag + creal
                imag = 2 * real * imag + cimag
                real = nreal
                if real * real + imag * imag > 4.0:
                    break
                n += 1
            out[j * grid_size + i] = i
            cimag += step_y
        creal += step_x
    return out


@njit
def tile_bounds(level, x, y, max_level, min_coord=-2.5, max_coord=2.5):
    max_width = max_coord - min_coord
    tile_width = max_width / 2 ** (max_level - level)
    from_x = min_coord + x * tile_width
    to_x = min_coord + (x + 1) * tile_width

    from_y = min_coord + y * tile_width
    to_y = min_coord + (y + 1) * tile_width

    return from_x, from_y, to_x, to_y


class MandlebrotStore(zarr.storage.Store):
    def __init__(self, levels, tilesize, maxiter=255, compressor=None):
        self.levels = levels
        self.tilesize = tilesize
        self.compressor = compressor
        self.dtype = np.dtype(np.uint8 if maxiter < 256 else np.uint16)
        self.maxiter = maxiter
        self._store = create_meta_store(
            levels, tilesize, compressor, self.dtype
        )

    def __getitem__(self, key):
        if key in self._store:
            return self._store[key]

        try:
            # Try parsing pyramidal coords
            level, chunk_key = key.split("/")
            level = int(level)
            y, x = map(int, chunk_key.split("."))
        except:
            raise KeyError

        from_x, from_y, to_x, to_y = tile_bounds(level, x, y, self.levels)
        out = np.zeros(self.tilesize * self.tilesize, dtype=self.dtype)
        #tile = mandelbrot(
        tile = xcoord_image(
                out, from_x, from_y, to_x, to_y, self.tilesize, self.maxiter
            )
        tile = tile.reshape(self.tilesize, self.tilesize).transpose()

        if self.compressor:
            return self.compressor.encode(tile)

        return tile.tobytes()

    def keys(self):
        return self._store.keys()

    def __iter__(self):
        return iter(self._store)

    def __delitem__(self, key):
        if key in self._store:
            del self._store[key]

    def __len__(self):
        return len(self._store)  # TODO not correct

    def __setitem__(self, key, val):
        self._store[key] = val


# https://dask.discourse.group/t/using-da-delayed-for-zarr-processing-memory-overhead-how-to-do-it-better/1007/10
def mandelbrot_dataset():
    max_levels = 8

    large_image = {
        "container": "mandelbrot.zarr/",
        "dataset": "",
        "scale_levels": max_levels,
        "scale_factors": [
            (2**level, 2**level) for level in range(max_levels)
        ],
        "chunk_size": (1024, 1024),
    }

    # Initialize the store
    store = MandlebrotStore(
        levels=max_levels, tilesize=512, compressor=Blosc()
    )
    # Wrap in a cache so that tiles don't need to be computed as often
    store = zarr.LRUStoreCache(store, max_size=1e9)

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
        arrays += [da.from_zarr(a, chunks=chunks)]

    large_image["arrays"] = arrays

    # TODO wrap in dask delayed

    return large_image
        
