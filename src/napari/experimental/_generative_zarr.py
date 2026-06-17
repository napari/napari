"""Generative zarr stores for testing progressive loading.

This module provides zarr v3 stores that synthesize multiscale image data
on demand: a 2D Mandelbrot set and a 3D Mandelbulb. They behave like
arbitrarily large pyramidal images without storing any chunk data — chunks
are computed when read, which makes them ideal stress tests for
progressive loading.

`numba <https://numba.pydata.org>`_ is used to accelerate chunk generation
when available, with a pure-Python fallback otherwise (slow, but
functional, and fine for tests with small tiles).
"""

from __future__ import annotations

import asyncio
import logging
import time

import numpy as np
import zarr
from zarr.abc.store import (
    ByteRequest,
    OffsetByteRequest,
    RangeByteRequest,
    SuffixByteRequest,
)
from zarr.core.buffer import Buffer, BufferPrototype, default_buffer_prototype
from zarr.storage import MemoryStore

LOGGER = logging.getLogger(__name__)

try:
    from numba import njit

    HAS_NUMBA = True
except ModuleNotFoundError:  # pragma: no cover - exercised without numba
    HAS_NUMBA = False

    def njit(*args, **kwargs):
        """No-op replacement for numba.njit."""

        def decorator(func):
            return func

        if args and callable(args[0]):
            return args[0]
        return decorator


@njit(nogil=True, cache=True)
def tile_bounds(level, coords, max_level, min_coord=-2.5, max_coord=2.5):
    """Return the fractal-space bounds of an ND tile.

    Parameters
    ----------
    level : int
        the scale level of this tile (0 is the highest resolution)
    coords : tuple/list
        a sequence of int tile coordinates
    max_level : int
        the maximum level of tiles
    min_coord : float
        the minimum coordinate of all tiles
    max_coord : float
        the maximum coordinate of all tiles

    """
    max_width = max_coord - min_coord
    tile_width = max_width / 2 ** (max_level - level)

    bounds = np.zeros((len(coords), 2))
    for idx, coord in enumerate(coords):
        start = min_coord + coord * tile_width
        stop = min_coord + (coord + 1) * tile_width
        bounds[idx, 0] = start
        bounds[idx, 1] = stop

    return bounds


@njit(nogil=True, fastmath=True, cache=True)
def mandelbrot(out, from_x, from_y, to_x, to_y, grid_size, maxiter):
    """Fill ``out`` with Mandelbrot escape iteration counts.

    ``out`` is a flat array of length ``grid_size ** 2`` filled in (row,
    col) = (y, x) C order.
    """
    step_x = (to_x - from_x) / grid_size
    step_y = (to_y - from_y) / grid_size
    for row in range(grid_size):
        cimag = from_y + row * step_y
        for col in range(grid_size):
            creal = from_x + col * step_x
            # Cardioid / period-2 bulb check for early termination
            q = (creal - 0.25) ** 2 + cimag * cimag
            if (
                q * (q + (creal - 0.25)) < 0.25 * cimag * cimag
                or (creal + 1.0) ** 2 + cimag * cimag < 0.0625
            ):
                out[row * grid_size + col] = maxiter
                continue
            real = 0.0
            imag = 0.0
            n = 0
            for _ in range(maxiter):
                nreal = real * real - imag * imag + creal
                imag = 2 * real * imag + cimag
                real = nreal
                if real * real + imag * imag > 4.0:
                    break
                n += 1
            out[row * grid_size + col] = n
    return out


# Based on http://www.fractal.org/Formula-Mandelbulb.pdf
@njit(nogil=True, fastmath=True, cache=True)
def mandelbulb(
    out, from_x, from_y, from_z, to_x, to_y, to_z, grid_size, maxiter, order,
):
    """Fill ``out`` with Mandelbulb escape iteration counts.

    ``out`` is a flat array of length ``grid_size ** 3`` filled in
    (z, y, x) C order.
    """
    step_x = (to_x - from_x) / grid_size
    step_y = (to_y - from_y) / grid_size
    step_z = (to_z - from_z) / grid_size

    for plane in range(grid_size):
        cz = from_z + plane * step_z
        for row in range(grid_size):
            cy = from_y + row * step_y
            for col in range(grid_size):
                cx = from_x + col * step_x
                x = 0.0
                y = 0.0
                z = 0.0
                n = 0
                for _ in range(maxiter):
                    # hypercomplex exponentiation
                    r = np.sqrt(x * x + y * y + z * z)
                    r_xy = np.sqrt(x * x + y * y)
                    theta = np.arctan2(z, r_xy)
                    phi = np.arctan2(y, x)
                    new_r = r**order
                    x = new_r * np.cos(order * phi) * np.cos(order * theta)
                    y = new_r * np.sin(order * phi) * np.cos(order * theta)
                    z = new_r * np.sin(order * theta)
                    x += cx
                    y += cy
                    z += cz
                    if x * x + y * y + z * z > 4.0:
                        break
                    n += 1
                out[plane * grid_size * grid_size + row * grid_size + col] = n
    return out


def _slice_buffer(data: bytes, byte_range: ByteRequest | None) -> bytes:
    """Apply a zarr ``ByteRequest`` to raw bytes."""
    if byte_range is None:
        return data
    if isinstance(byte_range, RangeByteRequest):
        return data[byte_range.start : byte_range.end]
    if isinstance(byte_range, OffsetByteRequest):
        return data[byte_range.offset :]
    if isinstance(byte_range, SuffixByteRequest):
        return data[-byte_range.suffix :]
    raise TypeError(f'Unexpected byte_range, got {byte_range}.')


class GenerativeZarrStore(MemoryStore):
    """A zarr v3 store whose chunks are computed on demand.

    Group/array metadata for a multiscale pyramid is created eagerly (and
    held in memory like a regular ``MemoryStore``), while chunk reads for
    keys like ``"<level>/c/<i>/<j>"`` are synthesized by
    :meth:`get_chunk`, which subclasses must implement.

    Note: every read recomputes the chunk. Wrap the store with
    ``zarr.experimental.cache_store.CacheStore`` to cache computed chunks.

    Parameters
    ----------
    levels : int
        Number of scale levels. Level 0 is the highest resolution with a
        width of ``tilesize * 2 ** levels``; each subsequent level halves
        the width.
    tilesize : int
        Chunk edge length used along every dimension.
    maxiter : int
        Maximum escape-time iterations; also determines the dtype
        (``uint8`` if it fits, ``uint16`` otherwise).
    ndim : int
        Number of dimensions of the generated arrays.
    cpu_relief : float
        After computing a chunk, sleep for this fraction of the CPU time
        the computation used. Chunk synthesis is pure compute, and many
        parallel readers can otherwise saturate every core and starve the
        GUI event loop. ``0`` disables pacing.

    """

    def __init__(
        self,
        levels: int,
        tilesize: int,
        maxiter: int = 255,
        *,
        ndim: int,
        cpu_relief: float = 0.5,
    ):
        super().__init__()
        self.levels = levels
        self.tilesize = tilesize
        self.maxiter = maxiter
        self.ndim = ndim
        self.cpu_relief = cpu_relief
        self.dtype = np.dtype(np.uint8 if maxiter < 256 else np.dtype('<u2'))
        self._init_metadata()

    def _init_metadata(self) -> None:
        root = zarr.create_group(store=self, zarr_format=3)
        datasets = [{'path': str(level)} for level in range(self.levels)]
        root.attrs['multiscales'] = [{'datasets': datasets, 'version': '0.1'}]
        base_width = self.tilesize * 2**self.levels
        for level in range(self.levels):
            width = base_width // 2**level
            root.create_array(
                name=str(level),
                shape=(width,) * self.ndim,
                chunks=(self.tilesize,) * self.ndim,
                dtype=self.dtype,
                compressors=None,
                fill_value=0,
            )

    def get_chunk(self, level: int, *coords: int) -> np.ndarray:
        """Compute the chunk at ``coords`` (C-order index order) of ``level``."""
        raise NotImplementedError

    def with_read_only(self, read_only: bool = True) -> GenerativeZarrStore:
        # MemoryStore.with_read_only reconstructs via __init__(store_dict=...),
        # which doesn't match this class's signature; share state via a
        # shallow copy instead (chunk synthesis is stateless).
        import copy

        new = copy.copy(self)
        new._read_only = read_only
        return new

    def _parse_chunk_key(self, key: str) -> tuple[int, ...] | None:
        parts = key.split('/')
        if len(parts) != self.ndim + 2 or parts[1] != 'c':
            return None
        try:
            level = int(parts[0])
            coords = tuple(int(part) for part in parts[2:])
        except ValueError:
            return None
        if not 0 <= level < self.levels:
            return None
        return (level, *coords)

    async def get(
        self,
        key: str,
        prototype: BufferPrototype | None = None,
        byte_range: ByteRequest | None = None,
    ) -> Buffer | None:
        existing = await super().get(key, prototype, byte_range)
        if existing is not None:
            return existing
        parsed = self._parse_chunk_key(key)
        if parsed is None:
            return None
        # run the (GIL-releasing) chunk computation off the event loop so
        # concurrent reads synthesize chunks in parallel

        def compute() -> tuple[np.ndarray, float]:
            cpu_start = time.thread_time()
            result = self.get_chunk(*parsed)
            return result, time.thread_time() - cpu_start

        chunk, cpu_used = await asyncio.to_thread(compute)
        if self.cpu_relief > 0 and cpu_used > 0:
            # leave the GUI thread some CPU between compute bursts
            await asyncio.sleep(cpu_used * self.cpu_relief)
        data = np.ascontiguousarray(chunk, dtype=self.dtype).tobytes()
        if prototype is None:
            prototype = default_buffer_prototype()
        return prototype.buffer.from_bytes(_slice_buffer(data, byte_range))


class MandelbrotStore(GenerativeZarrStore):
    """A multiscale zarr store generating the Mandelbrot set on demand."""

    def __init__(
        self, levels: int, tilesize: int, maxiter: int = 255, **kwargs,
    ):
        super().__init__(levels, tilesize, maxiter, ndim=2, **kwargs)

    def get_chunk(self, level: int, *coords: int) -> np.ndarray:
        y, x = coords
        bounds = tile_bounds(level, np.array([y, x]), self.levels)
        out = np.zeros(self.tilesize * self.tilesize, dtype=self.dtype)
        tile = mandelbrot(
            out,
            bounds[1, 0],
            bounds[0, 0],
            bounds[1, 1],
            bounds[0, 1],
            self.tilesize,
            self.maxiter,
        )
        return tile.reshape(self.tilesize, self.tilesize)


class MandelbulbStore(GenerativeZarrStore):
    """A multiscale zarr store generating a Mandelbulb on demand."""

    def __init__(
        self,
        levels: int,
        tilesize: int,
        maxiter: int = 255,
        order: int = 8,
        **kwargs,
    ):
        super().__init__(levels, tilesize, maxiter, ndim=3, **kwargs)
        self.order = order

    def get_chunk(self, level: int, *coords: int) -> np.ndarray:
        z, y, x = coords
        bounds = tile_bounds(
            level, np.array([z, y, x]), self.levels, -1.25, 1.25,
        )
        out = np.zeros(
            self.tilesize * self.tilesize * self.tilesize, dtype=self.dtype,
        )
        tile = mandelbulb(
            out,
            bounds[2, 0],
            bounds[1, 0],
            bounds[0, 0],
            bounds[2, 1],
            bounds[1, 1],
            bounds[0, 1],
            self.tilesize,
            self.maxiter,
            self.order,
        )
        return tile.reshape(self.tilesize, self.tilesize, self.tilesize)
