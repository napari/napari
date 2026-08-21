"""Histogram computation for IntensityMixin layers."""

from __future__ import annotations

import logging
import math
import warnings
from collections.abc import Generator, Sequence
from typing import Any, Literal

import numpy as np

from napari.utils._dask_utils import _is_dask_data

logger = logging.getLogger('napari.components.histogram')

# Maximum number of elements to materialize into a numpy array when
# the data is not chunked (e.g. h5py datasets).  Used as a safety
# guard in _get_full_data() — beyond this threshold we skip full-mode
# computation and warn instead of silently pulling the full array into
# memory.  ~50M float64 elements ≈ 400 MB.
_MAX_MATERIALIZE_ELEMENTS: int = 50_000_000


def compute_histogram(
    layer: Any,
    *,
    bins: int,
    max_samples: int,
    mode: Literal['canvas', 'full'],
    log_scale: bool,
) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
    """Yield ``(bin_edges, counts)`` for histogram computation.

    For chunked full-mode data, yields intermediate results after each
    chunk for progressive display. For non-chunked data, yields the final
    result once. Returns .
    """
    data = _get_data(layer, mode)

    if data is None or data.size == 0:
        empty = _make_empty()
        yield empty['bin_edges'], empty['counts']
        return

    if getattr(layer, 'rgb', False):
        data = _sample_rgb_and_luminance(data, max_samples)
        if data.size == 0:
            empty = _make_empty()
            yield empty['bin_edges'], empty['counts']
            return

    if mode == 'full' and _has_chunks(data):
        yield from _compute_chunked_progressive(
            data=data,
            contrast_limits_range=layer.contrast_limits_range,
            bins=bins,
            max_samples=max_samples,
            log_scale=log_scale,
        )

    else:
        if data.size > max_samples:
            data = _sample_data(data, max_samples)
        bin_edges, counts = _calc_histogram(data, layer, bins, log_scale)
        yield bin_edges, counts


def _make_empty():
    return {
        'bin_edges': np.array([0.0, 1.0]),
        'counts': np.array([0.0]),
    }


def _get_computed(layer: Any) -> dict[str, np.ndarray]:
    """Return the last computed histogram for *layer* (defaults if none)."""
    return layer.metadata.get('_computed_histogram', _make_empty())


def _calc_histogram(
    data: np.ndarray,
    layer: Any,
    bins: int,
    log_scale: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute bin edges and counts from data.

    Separates the pure numpy histogram computation from data fetching
    and preprocessing.

    Parameters
    ----------
    data : np.ndarray
        Preprocessed 1D data array.
    range_min : float
        Minimum value for histogram range.
    range_max : float
        Maximum value for histogram range.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (bin_edges, counts) — bin edges as float32, counts as float32.
    """
    range_min, range_max = layer.contrast_limits_range
    if range_min is None or range_max is None:
        range_min = float(np.nanmin(data))
        range_max = float(np.nanmax(data))

    # Handle edge case where min == max (constant data).
    # For integer types, ±0.5 places bin edges at half-integer
    # boundaries (e.g. uint8 value 42 → bin [41.5, 42.5]).
    # For float types, expand by 1 % of the value (min 0.5).
    if range_min == range_max:
        if np.issubdtype(layer.dtype, np.integer):
            range_min = float(range_min) - 0.5
            range_max = float(range_max) + 0.5
        else:
            delta = max(0.5, abs(range_min) * 0.01) if range_min != 0 else 0.5
            range_min = float(range_min) - delta
            range_max = float(range_max) + delta

    counts, bins = np.histogram(
        data,
        bins=bins,
        range=(float(range_min), float(range_max)),
    )

    bin_edges = bins.astype(np.float32)

    if log_scale:
        hist_counts = np.log10(counts + 1).astype(np.float32)
    else:
        hist_counts = counts.astype(np.float32)

    return bin_edges, hist_counts


def _rgb_to_luminance(data: np.ndarray) -> np.ndarray:
    """Convert RGB(A) data to perceptual luminance.

    Uses ITU-R BT.709 coefficients so the result matches sRGB display
    brightness. Only the first three channels are used; alpha is ignored.
    The returned array has the same value range as the input (e.g. 0-255
    for uint8, 0-1 for float).
    """
    rgb: np.ndarray = data[..., :3].astype(np.float32)
    return rgb @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def _sample_rgb_and_luminance(
    data: np.ndarray, max_samples: int
) -> np.ndarray:
    """Convert RGB(A) data to luminance, sampling pixels first for large data.

    For large RGB arrays, randomly samples ``max_samples`` pixel positions
    BEFORE converting to luminance to avoid materializing the full float32
    intermediate array. For small data, delegates to ``_rgb_to_luminance``.
    """
    n_pixels = data.size // data.shape[-1]
    if n_pixels <= max_samples:
        return _rgb_to_luminance(data)

    rng = np.random.default_rng(0)
    pixel_indices = rng.choice(n_pixels, size=max_samples, replace=False)
    pixel_indices.sort()  # sort for better dask graph contiguity
    # Flatten the spatial dimensions to (n_pixels, channels) and select
    # the sampled rows along a single axis.
    n_channels = data.shape[-1]
    flat = data.reshape(n_pixels, n_channels)
    sampled_rgb = np.asarray(flat[pixel_indices])
    luminance = _rgb_to_luminance(sampled_rgb)
    valid = np.isfinite(luminance)
    return luminance[valid]


def _get_data(
    layer: Any, mode: Literal['canvas', 'full']
) -> np.ndarray | None:
    """Get data from layer based on current mode."""
    if mode == 'canvas':
        return _get_displayed_data(layer)
    return _get_full_data(layer)


def _get_displayed_data(layer: Any) -> np.ndarray | None:
    """Get data from currently displayed slice.

    In 'canvas' mode, the histogram is computed from the visible data
    that has already been sliced for rendering. This uses the layer's
    ``_slice.image.raw`` which contains the data being displayed.

    Returns None if the slice is not yet available (e.g. during initial
    loading).
    """
    raw = _get_slice_raw_data(layer)
    if raw is not None and raw.size > 0:
        return raw
    return None


def _get_slice_raw_data(layer: Any) -> np.ndarray | None:
    """Get the currently sliced raw image data if available."""
    if type(layer).__name__ == 'Surface':
        data = layer._slicing_state._view_vertex_values
        return np.asarray(data) if data is not None else None

    layer_slice = layer._slice
    if layer_slice is None:
        return None
    raw = layer_slice.image.raw
    return np.asarray(raw) if raw is not None else None


def _get_full_data(layer: Any) -> np.ndarray | None:
    """Get full volume data, using coarsest level for multiscale."""
    if type(layer).__name__ == 'Surface':
        # Surface layers contain vertex values in the third element of the tuple.
        # Check if the surface has vortex values, if yes return them, otherwise return None.
        if len(layer.data) == 2:
            return None
        data = layer.data[2]
        return np.asarray(data)

    data = layer.data

    # Unpack multiscale to the coarsest level.
    if isinstance(data, Sequence) and not isinstance(
        data, (np.ndarray, str, bytes)
    ):
        data = data[-1]

    if isinstance(data, np.ndarray):
        return data

    # Chunked arrays (dask, zarr, h5py with chunks) are returned
    # as-is for the progressive sampler in _compute_chunked_progressive.
    if _has_chunks(data):
        return data

    # Last resort: cast to numpy.  Guard against accidentally
    # materializing a very large object (contiguous h5py) by
    # checking the estimated memory footprint first.
    data_size = data.size if hasattr(data, 'size') else 0
    if data_size > _MAX_MATERIALIZE_ELEMENTS:
        dtype_size = (
            np.dtype(data.dtype).itemsize if hasattr(data, 'dtype') else 8
        )
        est_mb = (data_size * dtype_size) / (1024 * 1024)
        warnings.warn(
            f'Skipping full-data histogram: materializing '
            f'{data_size:,} elements (~{est_mb:.0f} MB) would '
            f'exceed the safety limit of '
            f'{_MAX_MATERIALIZE_ELEMENTS:,} elements. '
            f'Use canvas mode or increase max_samples.',
            stacklevel=2,
        )
        return None
    return np.asarray(data)


def _has_chunks(data: Any) -> bool:
    """True if *data* can be sampled chunk-by-chunk (dask, zarr, h5py).

    h5py datasets all have a ``.chunks`` attribute, but it is None for
    unchunked/contiguous datasets:
    https://docs.h5py.org/en/latest/high/dataset.html#chunked-storage
    """
    if _is_dask_data(data):
        return True
    chunks = getattr(data, 'chunks', None)
    return chunks is not None and hasattr(data, 'shape')


def _chunk_sizes(data: Any) -> list[int]:
    """Return list of element counts for every chunk in *data*.

    Works for both dask (per-chunk tuple-of-tuples) and zarr
    (per-dimension scalar) arrays — only metadata is accessed.
    """
    import dask.array as da

    if isinstance(data, da.Array):
        nb, ch = data.numblocks, data.chunks
    else:
        # zarr: .chunks is (chunk_dim_0, chunk_dim_1, ...) not
        # per-chunk tuples. Compute block count from shape/chunks.
        nb = tuple(
            max(1, math.ceil(s / c))
            for s, c in zip(data.shape, data.chunks, strict=True)
        )
        ch = tuple(
            tuple(min(c, s - i * c) for i in range(n))
            for s, c, n in zip(data.shape, data.chunks, nb, strict=True)
        )

    sizes: list[int] = []
    for idx in np.ndindex(*nb):
        sz = 1
        for d, i in enumerate(idx):
            sz *= ch[d][i]
        sizes.append(sz)
    return sizes


def _load_chunk(data: Any, flat_idx: int) -> np.ndarray:
    """Load a single chunk by its flat index (dask or zarr)."""
    import dask.array as da

    if isinstance(data, da.Array):
        idx = np.unravel_index(flat_idx, data.numblocks)
        return np.asarray(data.blocks[idx]).ravel()

    # zarr path: convert flat chunk index to data-space slices
    nb = tuple(
        max(1, math.ceil(s / c))
        for s, c in zip(data.shape, data.chunks, strict=True)
    )
    idx = np.unravel_index(flat_idx, nb)
    slices: list[slice] = []
    for d, i in enumerate(idx):
        start = i * int(data.chunks[d])
        stop = min(start + int(data.chunks[d]), int(data.shape[d]))
        slices.append(slice(start, stop))
    return np.asarray(data[tuple(slices)]).ravel()


def _sample_data(data: np.ndarray, max_samples: int) -> np.ndarray:
    """Randomly sample data to reduce computation."""
    flat_data = data.ravel()
    valid_mask = np.isfinite(flat_data)
    valid_data = flat_data[valid_mask]

    if valid_data.size == 0:
        return np.array([])

    if valid_data.size <= max_samples:
        return valid_data

    rng = np.random.default_rng(0)
    indices = rng.choice(valid_data.size, size=max_samples, replace=False)
    return valid_data[indices]


def _compute_chunked_progressive(
    data: Any,
    contrast_limits_range: tuple[float | None, float | None],
    bins: int,
    max_samples: int,
    log_scale: bool,
) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
    """Generator that yields ``(bin_edges, counts)`` after each chunk.

    Provides incremental histogram snapshots as each chunk is loaded.
    """
    n = min(max_samples, data.size)
    chunk_sizes = _chunk_sizes(data)
    rng = np.random.default_rng()
    n_chunks = len(chunk_sizes)
    n_selected = min(n_chunks, max(1, n // max(1, min(chunk_sizes))))
    probs = np.asarray(chunk_sizes) / sum(chunk_sizes)
    order = rng.choice(n_chunks, size=n_selected, p=probs, replace=False)

    range_min, range_max = contrast_limits_range
    if range_min is None or range_max is None:
        range_min = 0.0
        range_max = 1.0

    running_counts = np.zeros(bins, dtype=np.float64)
    for ci in order:
        # Chunk load failure (e.g. remote zarr read error) is
        # non-fatal. Stop the generator instead of letting the
        # exception propagate through the GeneratorWorker's Qt
        # signal/slot machinery, which causes qFatal/abort on
        # PyQt6.
        try:
            block = _load_chunk(data, ci)
        except Exception:
            logger.warning('Histogram chunk load failed', exc_info=True)
            return
        chunk_counts, _ = np.histogram(
            block,
            bins=bins,
            range=(float(range_min), float(range_max)),
        )
        running_counts += chunk_counts.astype(np.float64)

        bins_arr = np.linspace(range_min, range_max, bins + 1).astype(
            np.float32
        )
        if log_scale:
            counts = np.log10(running_counts + 1).astype(np.float32)
        else:
            counts = running_counts.astype(np.float32)

        yield bins_arr, counts
