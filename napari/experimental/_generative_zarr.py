import logging
import sys

import numpy as np
import zarr
from numba import njit
from zarr.storage import init_array, init_group
from zarr.util import json_dumps

LOGGER = logging.getLogger("napari.experimental._generative_zarr")
LOGGER.setLevel(logging.DEBUG)

streamHandler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
streamHandler.setFormatter(formatter)
LOGGER.addHandler(streamHandler)


# Utilities to support generative zarrs
def create_meta_store(levels, tilesize, compressor, dtype, ndim=3):
    """Create metadata for a zarr Store.

    Parameters
    ----------
    levels : int
        Number of scale levels
    tilesize : int
        size of a tile, this will be used along all dimensions
    compressor : zarr compressor
        a zarr compressor, like blosc or None
    dtype : numpy dtype
        a numpy dtype
    ndim : int
        number of dimensions for the zarr

    """
    store = {}
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
            shape=tuple([width] * ndim),
            chunks=tuple([tilesize] * ndim),
            dtype=dtype,
            compressor=compressor,
        )
    return store


@njit(nogil=True)
def tile_bounds(level, coords, max_level, min_coord=-2.5, max_coord=2.5):
    """Return the bounds of a ND tile.

    Parameters
    ----------
    level : int
        the scale level of this tile
    coords : tuple/list
        a sequence of int coordinates
    max_level : int
        the maximum level of tiles
    min_coord : float
        the minimum coordinate of all tiles
    max_coord : float
        the maximum coordinate of all files
    """
    max_width = max_coord - min_coord
    tile_width = max_width / 2 ** (max_level - level)

    bounds = np.zeros((len(coords), 2))
    for idx, c in enumerate(coords):
        start = min_coord + c * tile_width
        stop = min_coord + (c + 1) * tile_width

        bounds[idx, :] = (start, stop)

    return bounds


# Mandelbrot
@njit(nogil=True)
def mandelbrot(out, from_x, from_y, to_x, to_y, grid_size, maxiter):
    """Return a 2D set of mandelbrot calculations.

    Parameters
    ----------
    out : ndarray-like
        a 2D ndarray
    from_x : int
        start coordinate along X-axis
    from_y : int
        start coordinate along Y-axis
    to_x : int
        end coordinate along X-axis
    to_y : int
        end coordinate along Y-axis
    grid_size : int
        the number of steps along any dimension
    maxiter : int
        the maximum number of iterations for calculating Mandelbrot

    """
    step_x = (to_x - from_x) / grid_size
    step_y = (to_y - from_y) / grid_size
    creal = from_x
    cimag = from_y
    for i in range(grid_size):
        cimag = from_y
        for j in range(grid_size):
            nreal = real = imag = n = 0
            # Use Cardioid / bulb checking for early termination
            q = (i - 0.25) ** 2 + j**2
            if q * (q + (i - 0.25)) > 0.25 * j**2:
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


class MandelbrotStore(zarr.storage.Store):
    """A Zarr store for generating the Mandelbrot set."""

    def __init__(self, levels, tilesize, maxiter=255, compressor=None):
        self.levels = levels
        self.tilesize = tilesize
        self.compressor = compressor
        self.dtype = np.dtype(np.uint8 if maxiter < 256 else np.uint16)
        self.maxiter = maxiter
        self._store = create_meta_store(
            levels, tilesize, compressor, self.dtype, ndim=2
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

        return self.get_chunk(level, y, x).tobytes()

    def get_chunk(self, level, y, x):
        bounds = tile_bounds(level, (x, y), self.levels)
        from_x, from_y = bounds[:, 0]
        to_x, to_y = bounds[:, 1]
        out = np.zeros(self.tilesize * self.tilesize, dtype=self.dtype)
        tile = mandelbrot(
            # tile = xcoord_image(
            out,
            from_x,
            from_y,
            to_x,
            to_y,
            self.tilesize,
            self.maxiter,
        )
        tile = tile.reshape(self.tilesize, self.tilesize).transpose()

        if self.compressor:
            return self.compressor.encode(tile)

        return tile

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


# Mandelbulb
# Based on http://www.fractal.org/Formula-Mandelbulb.pdf
@njit(nogil=True)
def hypercomplex_exponentiation(x, y, z, n):
    """Hypercomplex exponentiation to transform coordinates in the Mandelbulb.

    Parameters
    ----------
    x : float
        a X-coordinate
    y : float
        a Y-coordinate
    z : float
        a Z-coordinate
    n : int
        the order of the Mandelbulb

    """
    r = np.sqrt(x * x + y * y + z * z)
    r1 = np.sqrt(x * x + y * y)
    theta = np.arctan2(z, r1)
    phi = np.arctan2(y, x)
    new_r = r**n
    new_x = new_r * np.cos(n * phi) * np.cos(n * theta)
    new_y = new_r * np.sin(n * phi) * np.cos(n * theta)
    new_z = new_r * np.sin(n * theta)
    return new_x, new_y, new_z


@njit(nogil=True)
def mandelbulb(
    out, from_x, from_y, from_z, to_x, to_y, to_z, grid_size, maxiter, n
):
    """Compute the Mandelbulb.

    Parameters
    ----------
    out : ndarray-like
        a 3D array for storing the Mandelbulb results
    from_x : int
        the start X-coordinate
    from_y : int
        the start Y-coordinate
    from_z : int
        the start Z-coordinate
    to_x : int
        the end X-coordinate
    to_y : int
        the end Y-coordinate
    to_z : int
        the end Z-coordinate
    grid_size : int
        the number of steps to take along each dimensions
    maxiter : int
        the maximum number of iterations for calculating the
        Mandelbulb
    n : int
        the order of the Mandelbulb equation

    """
    step_x = (to_x - from_x) / grid_size
    step_y = (to_y - from_y) / grid_size
    step_z = (to_z - from_z) / grid_size

    for k in range(grid_size):  # Z loop
        cimag2 = from_z + k * step_z
        for j in range(grid_size):  # Y loop
            cimag = from_y + j * step_y
            for i in range(grid_size):  # X loop
                creal = from_x + i * step_x
                nreal = real = imag = imag2 = n_iter = 0
                for _ in range(maxiter):
                    nreal, nimag, nimag2 = hypercomplex_exponentiation(
                        real, imag, imag2, n
                    )
                    nreal += creal
                    nimag += cimag
                    nimag2 += cimag2
                    real = nreal
                    imag = nimag
                    imag2 = nimag2
                    if real * real + imag * imag + imag2 * imag2 > 4.0:
                        break
                    out[k * grid_size * grid_size + j * grid_size + i] = n_iter
                    n_iter += 1

    return out


class MandelbulbStore(zarr.storage.Store):
    """A Zarr store for generating the Mandelbulb."""

    def __init__(
        self, levels, tilesize, maxiter=255, compressor=None, order=4
    ):
        self.levels = levels
        self.tilesize = tilesize
        self.compressor = compressor
        self.dtype = np.dtype(np.uint8 if maxiter < 256 else np.uint16)
        self.maxiter = maxiter
        self.order = order
        self._store = create_meta_store(
            levels, tilesize, compressor, self.dtype, ndim=3
        )

    def __getitem__(self, key):
        if key in self._store:
            return self._store[key]

        try:
            # Try parsing pyramidal coords
            level, chunk_key = key.split("/")
            level = int(level)
            z, y, x = map(int, chunk_key.split("."))
        except:
            raise KeyError

        return self.get_chunk(level, z, y, x).tobytes()

    def get_chunk(self, level, z, y, x):
        bounds = tile_bounds(level, (x, y, z), self.levels)
        from_x, from_y, from_z = bounds[:, 0]
        to_x, to_y, to_z = bounds[:, 1]
        out = np.zeros(
            self.tilesize * self.tilesize * self.tilesize, dtype=self.dtype
        )
        tile = mandelbulb(
            out,
            from_x,
            from_y,
            from_z,
            to_x,
            to_y,
            to_z,
            self.tilesize,
            self.maxiter,
            self.order,
        )
        tile = tile.reshape(
            self.tilesize, self.tilesize, self.tilesize
        ).transpose()

        if self.compressor:
            return self.compressor.encode(tile)

        return tile

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
