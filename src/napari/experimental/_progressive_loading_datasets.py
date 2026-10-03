"""Example multiscale datasets for progressive loading.

Each function returns a dictionary describing a multiscale image with (at
least) the keys ``'arrays'`` (list of array-like levels, highest resolution
first) and ``'scale_factors'``. The ``'arrays'`` entry can be passed
directly to
:func:`napari.experimental._progressive_loading.add_progressive_loading_image`.

The generative datasets (:func:`mandelbrot_dataset`,
:func:`mandelbulb_dataset`) have no dependencies beyond zarr (numba is
used when available). The remote datasets require optional extras noted in
their docstrings and are imported lazily.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import zarr
from zarr.experimental.cache_store import CacheStore
from zarr.storage import LocalStore, MemoryStore

from napari.experimental._generative_zarr import (
    MandelbrotStore,
    MandelbulbStore,
)

LOGGER = logging.getLogger(__name__)

#: Default size of the in-memory chunk cache for generative datasets.
DEFAULT_CACHE_BYTES = int(4e9)


def _open_cached_multiscale(store, levels: int, cache_bytes: int):
    """Open a multiscale group through an in-memory chunk cache."""
    cached = CacheStore(store, cache_store=MemoryStore(), max_size=cache_bytes)
    group = zarr.open_group(cached, mode='r')
    return [group[str(level)] for level in range(levels)]


def _is_remote(path: str) -> bool:
    return str(path).startswith(('http://', 'https://', 's3://', 'gs://'))


def _find_multiscales(group, depth: int = 0):
    """Walk (up to a few levels) into nested groups for OME multiscales.

    Returns ``(group, multiscales_dict)`` for the first group whose attrs
    carry a ``multiscales`` list (v0.1-v0.4 or the v0.5 ``ome`` wrapper),
    or ``(None, None)`` if none is found within ``depth`` levels.
    """
    attrs = dict(group.attrs)
    ome = attrs.get('ome')
    ms_list = attrs.get('multiscales') or (
        ome.get('multiscales') if isinstance(ome, dict) else None
    )
    if ms_list:
        return group, ms_list[0]
    if depth > 4:
        return None, None
    for key in group:
        try:
            child = group[key]
        except (KeyError, ValueError, OSError):
            continue
        # only groups can hold multiscales metadata; recursing into an
        # array would iterate its rows and fancy-index it (BoundsCheckError)
        if not isinstance(child, zarr.Group):
            continue
        found, ms = _find_multiscales(child, depth + 1)
        if ms is not None:
            return found, ms
    return None, None


def open_ome_zarr(
    path: str,
    *,
    num_levels: int | None = None,
    cache_bytes: int = DEFAULT_CACHE_BYTES,
    zarr_format: int | None = None,
    anon: bool = True,
    squeeze: bool = True,
):
    """Open a local or remote OME-Zarr for progressive loading.

    Consolidates the store setup and OME metadata parsing shared by the
    progressive-loading examples: remote URLs (``http(s)://``, ``s3://``,
    ``gs://``) are read through an in-memory :class:`CacheStore`, local
    paths through a :class:`LocalStore`. The first ``multiscales`` group
    reachable from the root is used.

    Parameters
    ----------
    path : str
        Local filesystem path or remote URL to the OME-Zarr group.
    num_levels : int, optional
        Keep only the first ``num_levels`` resolution levels (default: all).
    cache_bytes : int
        Size of the in-memory chunk cache for remote stores.
    zarr_format : int, optional
        Force the zarr format (2 or 3) when auto-detection is ambiguous.
    anon : bool
        Use anonymous access for ``s3://`` URLs (default True).
    squeeze : bool
        Drop leading singleton dimensions (e.g. 5D ``(1,1,1,H,W)`` IDR
        data) down to the spatial axes, adjusting ``scale``/``translate``
        to match.

    Returns
    -------
    arrays : list
        Multiscale levels, highest resolution first.
    scale : list of float or None
        Per-axis scale from the level-0 coordinate transformations.
    translate : list of float or None
        Per-axis translation from the level-0 coordinate transformations.
    """
    if _is_remote(path):
        from zarr.storage import FsspecStore

        storage_options = {'anon': anon} if path.startswith('s3://') else {}
        store = CacheStore(
            FsspecStore.from_url(path, storage_options=storage_options),
            cache_store=MemoryStore(),
            max_size=cache_bytes,
        )
    else:
        store = LocalStore(str(path))

    open_kw = {'mode': 'r'}
    if zarr_format is not None:
        open_kw['zarr_format'] = zarr_format
    root = zarr.open_group(store, **open_kw)

    group, ms = _find_multiscales(root)
    datasets = ms.get('datasets', []) if ms is not None else []
    if not datasets:
        # no multiscales metadata (or an empty datasets list): treat the
        # sorted child arrays of the group as the resolution levels
        group = group if group is not None else root
        datasets = [
            {'path': k}
            for k in sorted(group.keys(), key=lambda k: (len(k), k))
        ]
    if num_levels is not None:
        datasets = datasets[:num_levels]
    arrays = [group[ds['path']] for ds in datasets]

    scale, translate = None, None
    if datasets:
        for t in datasets[0].get('coordinateTransformations', []):
            if t.get('type') == 'scale':
                scale = list(t['scale'])
            elif t.get('type') == 'translation':
                translate = list(t['translation'])

    if squeeze and arrays and arrays[0].ndim > 3:
        import dask.array as da

        # collapse only LEADING singleton axes (the t/c axes of 5D
        # OME-Zarr, e.g. (1,1,1,H,W)) down toward the spatial axes;
        # interior/trailing singletons are kept so the scale/translate
        # trim below (which drops leading entries) stays axis-aligned
        shape0 = arrays[0].shape
        n_lead = 0
        while n_lead < len(shape0) - 2 and shape0[n_lead] == 1:
            n_lead += 1
        if n_lead:
            collapse = (0,) * n_lead
            arrays = [da.from_zarr(a)[collapse] for a in arrays]

    # trim scale/translate to the (possibly squeezed) dimensionality
    ndim = arrays[0].ndim if arrays else None
    if ndim is not None:
        if scale is not None and len(scale) > ndim:
            scale = scale[-ndim:]
        if translate is not None and len(translate) > ndim:
            translate = translate[-ndim:]

    return arrays, scale, translate


def mandelbrot_dataset(
    max_levels: int = 14,
    tilesize: int = 512,
    maxiter: int = 255,
    cache_bytes: int = DEFAULT_CACHE_BYTES,
    cpu_relief: float = 0.5,
):
    """Generate a multiscale 2D image of the Mandelbrot set.

    Scale 0 is the highest resolution (``tilesize * 2 ** max_levels``
    pixels wide); chunks are computed on demand and cached in memory.

    >>> large_image = mandelbrot_dataset(max_levels=14)
    >>> add_progressive_loading_image(large_image['arrays'], viewer=viewer)

    Parameters
    ----------
    max_levels : int
        Number of levels (scales) to generate.
    tilesize : int
        Chunk edge length.
    maxiter : int
        Maximum escape-time iterations (determines dtype).
    cache_bytes : int
        Size of the in-memory chunk cache.

    Returns
    -------
    dict
        Multiscale metadata with keys ``['container', 'dataset',
        'scale_levels', 'scale_factors', 'chunk_size', 'arrays']``.

    """
    store = MandelbrotStore(
        levels=max_levels,
        tilesize=tilesize,
        maxiter=maxiter,
        cpu_relief=cpu_relief,
    )
    arrays = _open_cached_multiscale(store, max_levels, cache_bytes)
    return {
        'container': 'mandelbrot.zarr/',
        'dataset': '',
        'scale_levels': max_levels,
        'scale_factors': [(2**level, 2**level) for level in range(max_levels)],
        'chunk_size': (tilesize, tilesize),
        'arrays': arrays,
    }


def mandelbulb_dataset(
    max_levels: int = 6,
    tilesize: int = 32,
    maxiter: int = 255,
    order: int = 8,
    cache_bytes: int = DEFAULT_CACHE_BYTES,
    cpu_relief: float = 0.5,
):
    """Generate a multiscale 3D image of a Mandelbulb.

    Parameters
    ----------
    max_levels : int
        Number of levels (scales) to generate.
    tilesize : int
        Chunk edge length (chunks are ``tilesize**3`` voxels).
    maxiter : int
        Maximum escape-time iterations (determines dtype).
    order : int
        Order of the Mandelbulb equation.
    cache_bytes : int
        Size of the in-memory chunk cache.

    Returns
    -------
    dict
        Multiscale metadata with keys ``['container', 'dataset',
        'scale_levels', 'scale_factors', 'chunk_size', 'arrays']``.

    """
    store = MandelbulbStore(
        levels=max_levels,
        tilesize=tilesize,
        maxiter=maxiter,
        order=order,
        cpu_relief=cpu_relief,
    )
    arrays = _open_cached_multiscale(store, max_levels, cache_bytes)
    return {
        'container': 'mandelbulb.zarr/',
        'dataset': '',
        'scale_levels': max_levels,
        'scale_factors': [
            (2**level, 2**level, 2**level) for level in range(max_levels)
        ],
        'chunk_size': (tilesize, tilesize, tilesize),
        'arrays': arrays,
    }


def _escape_to_rgb(chunk, maxiter, dtype=np.uint8):
    """Map scalar Mandelbulb escape-time values to an RGB triplet.

    The triplet holds 0-255 colour values in ``dtype`` (the store's own
    dtype: ``uint8``, or ``uint16`` when ``maxiter >= 256``), so the
    serialized chunk bytes match the array's declared dtype.
    """
    norm = chunk.astype(np.float32) / maxiter
    r = (np.sin(2 * np.pi * norm * 3.0) * 127 + 128).astype(dtype)
    g = (np.sin(2 * np.pi * norm * 5.0 + 2.0) * 127 + 128).astype(dtype)
    b = (np.sin(2 * np.pi * norm * 7.0 + 4.0) * 127 + 128).astype(dtype)
    mask = chunk >= maxiter
    r[mask] = g[mask] = b[mask] = 0
    return np.stack([r, g, b], axis=-1)


class MandelbulbRGBStore(MandelbulbStore):
    """Mandelbulb store producing ``(Z, Y, X, 3)`` RGB chunks lazily.

    Reuses :class:`MandelbulbStore`'s escape-time generation and maps each
    scalar value to an RGB triplet, so the same volume renders as a colour
    field through the 3D progressive-loading RGB texture path. Chunks use
    the store's ``dtype`` (``uint8``, or ``uint16`` for ``maxiter >= 256``).
    """

    def _init_metadata(self):
        root = zarr.create_group(store=self, zarr_format=3)
        datasets = [{'path': str(level)} for level in range(self.levels)]
        root.attrs['multiscales'] = [{'datasets': datasets, 'version': '0.1'}]
        base_width = self.tilesize * 2**self.levels
        for level in range(self.levels):
            width = base_width // 2**level
            root.create_array(
                name=str(level),
                shape=(width, width, width, 3),
                chunks=(self.tilesize, self.tilesize, self.tilesize, 3),
                dtype=self.dtype,
                compressors=None,
                fill_value=0,
            )

    def _parse_chunk_key(self, key):
        parts = key.split('/')
        # the RGB arrays carry one extra (channel) axis beyond the base
        # scalar store, so the key has one more coordinate than usual:
        # level/c/<ndim spatial coords>/rgb_idx
        if len(parts) != self.ndim + 3 or parts[1] != 'c':
            return None
        try:
            level = int(parts[0])
            coords = tuple(int(p) for p in parts[2:])
        except ValueError:
            return None
        if not 0 <= level < self.levels:
            return None
        return (level, *coords)

    def get_chunk(self, level, *coords):
        z, y, x, _rgb = coords
        scalar = super().get_chunk(level, z, y, x)
        return _escape_to_rgb(scalar, self.maxiter, self.dtype)


def mandelbulb_rgb_dataset(
    max_levels: int = 5,
    tilesize: int = 32,
    maxiter: int = 64,
    order: int = 8,
    cache_bytes: int = DEFAULT_CACHE_BYTES,
    cpu_relief: float = 0.5,
):
    """Generate a multiscale 3D *RGB* Mandelbulb.

    Like :func:`mandelbulb_dataset` but each voxel is an ``(R, G, B)``
    uint8 triplet, exercising the RGB texture path in 3D progressive
    loading. Chunks are generated on first access.

    Returns
    -------
    dict
        Multiscale metadata with the same keys as
        :func:`mandelbulb_dataset`; ``'arrays'`` levels have a trailing
        length-3 channel axis.
    """
    store = MandelbulbRGBStore(
        levels=max_levels,
        tilesize=tilesize,
        maxiter=maxiter,
        order=order,
        cpu_relief=cpu_relief,
    )
    arrays = _open_cached_multiscale(store, max_levels, cache_bytes)
    return {
        'container': 'mandelbulb_rgb.zarr/',
        'dataset': '',
        'scale_levels': max_levels,
        'scale_factors': [
            (2**level, 2**level, 2**level) for level in range(max_levels)
        ],
        'chunk_size': (tilesize, tilesize, tilesize, 3),
        'arrays': arrays,
    }


def build_multiscale_zarr(
    path: str | Path,
    max_levels: int = 4,
    tilesize: int = 32,
    maxiter: int = 64,
    order: int = 8,
    overwrite: bool = False,
) -> Path:
    """Materialize a multiscale mandelbulb zarr to disk.

    Generates a 3D mandelbulb pyramid and writes it chunk-by-chunk to a
    local zarr store. With the defaults (4 levels, tilesize 32) the
    result is ~150 MB on disk (512**3 + 256**3 + ... uint8).

    Parameters
    ----------
    path : str or Path
        Directory to write the zarr store into.
    max_levels : int
        Number of pyramid levels (level 0 = tilesize * 2**max_levels
        voxels per side).
    tilesize : int
        Chunk edge length.
    maxiter : int
        Mandelbulb escape iterations (determines intensity range).
    order : int
        Order of the mandelbulb equation.
    overwrite : bool
        If *True*, remove an existing store at *path* first.

    Returns
    -------
    Path
        The store directory (same as *path*).
    """
    path = Path(path)
    if path.exists() and not overwrite:
        LOGGER.info('Zarr already exists at %s, skipping build', path)
        return path

    gen = MandelbulbStore(
        levels=max_levels,
        tilesize=tilesize,
        maxiter=maxiter,
        order=order,
        cpu_relief=0,
    )

    store = LocalStore(path)
    root = zarr.create_group(store=store, zarr_format=3, overwrite=overwrite)
    datasets = [{'path': str(lvl)} for lvl in range(max_levels)]
    root.attrs['multiscales'] = [{'datasets': datasets, 'version': '0.1'}]

    base_width = tilesize * 2**max_levels
    for level in range(max_levels):
        width = base_width // 2**level
        n_chunks = width // tilesize
        arr = root.create_array(
            name=str(level),
            shape=(width, width, width),
            chunks=(tilesize, tilesize, tilesize),
            dtype=np.uint8 if maxiter < 256 else np.dtype('<u2'),
            compressors=None,
            fill_value=0,
            overwrite=overwrite,
        )
        total = n_chunks**3
        for zi in range(n_chunks):
            for yi in range(n_chunks):
                for xi in range(n_chunks):
                    chunk = gen.get_chunk(level, zi, yi, xi)
                    s = np.s_[
                        zi * tilesize : (zi + 1) * tilesize,
                        yi * tilesize : (yi + 1) * tilesize,
                        xi * tilesize : (xi + 1) * tilesize,
                    ]
                    arr[s] = chunk
                    done = zi * n_chunks * n_chunks + yi * n_chunks + xi + 1
                    if done % max(1, total // 10) == 0 or done == total:
                        LOGGER.info(
                            'Level %d: %d/%d chunks', level, done, total
                        )
    LOGGER.info('Wrote %s', path)
    return path


def local_zarr_dataset(
    path: str | Path,
    cache_bytes: int = DEFAULT_CACHE_BYTES,
):
    """Open a local multiscale zarr for progressive loading.

    If the zarr does not exist yet, builds a default mandelbulb first
    via :func:`build_multiscale_zarr`.

    Parameters
    ----------
    path : str or Path
        Path to the zarr store directory.
    cache_bytes : int
        Size of the in-memory chunk cache.

    Returns
    -------
    dict
        Multiscale metadata with ``'arrays'`` ready for
        :func:`add_progressive_loading_image`.
    """
    path = Path(path)
    if not path.exists():
        build_multiscale_zarr(path)

    store = CacheStore(
        LocalStore(path),
        cache_store=MemoryStore(),
        max_size=cache_bytes,
    )
    group = zarr.open_group(store, mode='r')
    multiscales = group.attrs.get('multiscales', [{}])
    level_paths = [d['path'] for d in multiscales[0].get('datasets', [])]
    if not level_paths:
        level_paths = sorted(k for k in group if k.isdigit())

    arrays = [group[p] for p in level_paths]
    ndim = arrays[0].ndim
    scale_factors = [
        tuple(2**level for _ in range(ndim)) for level in range(len(arrays))
    ]
    return {
        'container': str(path),
        'dataset': '',
        'scale_levels': len(arrays),
        'scale_factors': scale_factors,
        'chunk_size': arrays[0].chunks,
        'arrays': arrays,
    }


def openorganelle_mouse_kidney_em():
    """Mouse kidney FIB-SEM data from OpenOrganelle (remote, ~TB scale).

    Requires the optional ``fibsem_tools`` package.
    """
    try:
        from fibsem_tools import read_xarray
    except ModuleNotFoundError as e:  # pragma: no cover - optional dep
        raise ModuleNotFoundError(
            'openorganelle_mouse_kidney_em requires fibsem_tools: '
            'pip install fibsem_tools',
        ) from e

    large_image = {
        'container': 's3://janelia-cosem-datasets/jrc_mus-kidney/jrc_mus-kidney.n5',
        'dataset': 'em/fibsem-uint8',
        'scale_levels': 5,
        'scale_factors': [
            (1, 1, 1),
            (2, 2, 2),
            (4, 4, 4),
            (8, 8, 8),
            (16, 16, 16),
        ],
    }
    large_image['arrays'] = [
        read_xarray(
            f'{large_image["container"]}/{large_image["dataset"]}/s{scale}/',
            storage_options={'anon': True},
        ).data
        for scale in range(large_image['scale_levels'])
    ]
    return large_image


def luethi_zenodo_7144919(cache_bytes: int = DEFAULT_CACHE_BYTES):
    """Multiscale OME-Zarr of cardiomyocyte differentiation (Zenodo 7144919).

    Downloads ~600 MB on first use (cached by pooch). Requires the optional
    ``pooch`` package.
    """
    import os

    try:
        import pooch
    except ModuleNotFoundError as e:  # pragma: no cover - optional dep
        raise ModuleNotFoundError(
            'luethi_zenodo_7144919 requires pooch: pip install pooch',
        ) from e

    # Downloaded from https://zenodo.org/record/7144919
    dest_dir = pooch.retrieve(
        url='https://zenodo.org/record/7144919/files/20200812-CardiomyocyteDifferentiation14-Cycle1.zarr.zip?download=1',
        known_hash='e6773fc97dcf3689e2f42e6504e0d4f4d0845c329dfbdfe92f61c2f3f1a4d55d',
        processor=pooch.Unzip(),
    )
    local_container = os.path.split(dest_dir[0])[0]

    large_image = {
        'container': local_container,
        'dataset': 'B/03/0',
        'scale_levels': 5,
        'scale_factors': [
            (1, 0.1625, 0.1625),
            (1, 0.325, 0.325),
            (1, 0.65, 0.65),
            (1, 1.3, 1.3),
            (1, 2.6, 2.6),
        ],
        'chunk_size': (1, 10, 256, 256),
    }

    store = CacheStore(
        zarr.storage.LocalStore(local_container),
        cache_store=MemoryStore(),
        max_size=cache_bytes,
    )
    group = zarr.open_group(store, mode='r')
    multiscale_data = group[large_image['dataset']]
    large_image['arrays'] = [
        multiscale_data[str(scale)]
        for scale in range(large_image['scale_levels'])
    ]
    return large_image
