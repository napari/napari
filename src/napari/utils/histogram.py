"""Histogram computation for IntensityMixin layers."""

from __future__ import annotations

import itertools
import logging
import math
import threading
import warnings
import weakref
from collections.abc import Generator, Sequence
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from napari.types import ArrayLike
from napari.utils._dask_utils import _is_dask_data

if TYPE_CHECKING:
    from napari.layers import Image, Surface

if TYPE_CHECKING:
    from napari.layers.intensity_mixin import IntensityVisualizationMixin


logger = logging.getLogger('napari.components.histogram')

# Maximum number of elements to materialize into a numpy array when
# the data is not chunked (e.g. h5py datasets).  Used as a safety
# guard in _get_full_data() — beyond this threshold we skip full-mode
# computation and warn instead of silently pulling the full array into
# memory.  ~50M float64 elements ≈ 400 MB.
_MAX_MATERIALIZE_ELEMENTS: int = 50_000_000

VALUES_KEY = '_histogram_values'
FINGERPRINT_KEY = '_histogram_fingerprint'

# thread-safe caching machinery
_lock = threading.RLock()
# maps id(obj) to (token, weakref to obj) so that a token stays valid for as
# long as the object it identifies is alive
_token_counter = itertools.count(1)
_tokens: dict[int, tuple[int, weakref.ref]] = {}
_MAX_TOKENS = 1024


def compute_histogram(
    layer: IntensityVisualizationMixin,
    *,
    bin_count: int,
    max_samples: int,
    mode: Literal['canvas', 'full'],
    log_scale: bool,
) -> Generator[
    tuple[
        np.ndarray[tuple[int], np.dtype[np.float32]],
        np.ndarray[tuple[int], np.dtype[np.float32]],
    ],
    None,
    None,
]:
    """Yield ``(bin_edges, counts)`` for histogram computation.

    For chunked full-mode data, yields intermediate results after each
    chunk for progressive display. For non-chunked data, yields the final
    result once.

    Parameters
    ----------
    layer : IntensityVisualizationMixin
        The napari layer for which to compute the histogram (must be
        an intensity visualisation layer such as Image or Surface).
    bin_count : int
        Number of bins for the histogram.
    max_samples : int
        Maximum number of samples to use for histogram computation.
    mode : Literal['canvas', 'full']
        Whether to compute the histogram from the currently displayed
        slice ('canvas') or the full data ('full').
    log_scale : bool
        Whether to use a logarithmic scale for the histogram.

    Yields
    ------
    tuple[np.ndarray, np.ndarray]
        A tuple containing the bin edges and counts for the histogram.
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
            bin_count=bin_count,
            max_samples=max_samples,
            log_scale=log_scale,
        )

    else:
        if data.size > max_samples:
            data = _sample_data(data, max_samples)
        bin_edges, counts = _calc_histogram(data, layer, bin_count, log_scale)
        yield bin_edges, counts


def _make_empty() -> dict[str, np.ndarray]:
    return {
        'bin_edges': np.array([0.0, 1.0]),
        'counts': np.array([0.0]),
    }


def _calc_histogram(
    data: np.ndarray,
    layer: IntensityVisualizationMixin,
    bin_count: int,
    log_scale: bool,
) -> tuple[
    np.ndarray[tuple[int], np.dtype[np.float32]],
    np.ndarray[tuple[int], np.dtype[np.float32]],
]:
    """Compute bin edges and counts from data.

    Separates the pure numpy histogram computation from data fetching
    and preprocessing.

    Parameters
    ----------
    data : np.ndarray
        Preprocessed 1D data array.
    layer : IntensityVisualizationMixin
        The napari layer for which to compute the histogram.
    bin_count : int
        Number of bins for the histogram.
    log_scale : bool
        Whether to use a logarithmic scale for the histogram.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (bin_edges, counts) — bin edges as float32, counts as float32.
    """
    range_min, range_max = layer.contrast_limits_range

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

    counts, bin_edges = np.histogram(
        data,
        bins=bin_count,
        range=(float(range_min), float(range_max)),
    )

    assert bin_edges.ndim == 1
    assert counts.ndim == 1

    hist_counts = np.log10(counts + 1) if log_scale else counts

    # Convert to float32 for consistency and to reduce memory usage.
    return bin_edges.astype(np.float32), hist_counts.astype(np.float32)


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


def _canvas_source(layer: IntensityVisualizationMixin) -> Any | None:
    """Return the object the canvas-mode histogram is computed from."""
    from napari.layers import Image, Surface

    if not hasattr(layer, '_slicing_state'):
        # layer not yet fully initialized
        return None
    if isinstance(layer, Surface):
        return layer._slicing_state._view_vertex_values
    if isinstance(layer, Image):
        return layer._slicing_state._slice.image.raw

    raise NotImplementedError(
        f'_canvas_source not implemented for layer type {layer.__class__.__name__}'
    )


def _full_source(layer: IntensityVisualizationMixin) -> Any | None:
    """Return the object the full-mode histogram is computed from."""
    from napari.layers import Image, Surface

    if isinstance(layer, Surface):
        # surface layers hold the vertex values in the third element
        return layer.data[2] if len(layer.data) == 3 else None
    if isinstance(layer, Image):
        data = layer.data
        if isinstance(data, Sequence) and not isinstance(
            data, (np.ndarray, str, bytes)
        ):
            # multiscale data: the coarsest level is used
            data = data[-1]
        return data

    raise NotImplementedError(
        f'_full_source not implemented for layer type {layer.__class__.__name__}'
    )


def _get_data(layer: Any, mode: Literal['canvas', 'full']) -> ArrayLike | None:
    """Get data from layer based on current mode."""
    source_data = (
        _canvas_source(layer) if mode == 'canvas' else _full_source(layer)
    )
    data = _to_array(source_data)
    return data if data is not None and data.size > 0 else None


def _to_array(source: Any) -> ArrayLike | None:
    """Convert (if needed) the source to a usable array, respecting lazy/chuncked data."""
    if isinstance(source, np.ndarray):
        return source

    # Chunked arrays (dask, zarr, h5py with chunks) are returned
    # as-is for the progressive sampler in _compute_chunked_progressive.
    if _has_chunks(source):
        return source

    # Last resort: cast to numpy.  Guard against accidentally
    # materializing a very large object (contiguous h5py) by
    # checking the estimated memory footprint first.
    data_size = source.size if hasattr(source, 'size') else 0
    if data_size > _MAX_MATERIALIZE_ELEMENTS:
        dtype_size = (
            np.dtype(source.dtype).itemsize if hasattr(source, 'dtype') else 8
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
    return np.asarray(source)


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
    contrast_limits_range: tuple[float, float],
    bin_count: int,
    max_samples: int,
    log_scale: bool,
) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
    """Generator that yields ``(bin_edges, counts)`` after each chunk.

    Provides incremental histogram snapshots as each chunk is loaded.

    Parameters
    ----------
    data : Any
        The chunked data array (dask or zarr).
    contrast_limits_range : tuple[float, float]
        The contrast limits range for histogram computation.
    bin_count : int
        Number of bins for the histogram.
    max_samples : int
        Maximum number of samples to use for histogram computation.
    log_scale : bool
        Whether to use a logarithmic scale for the histogram.

    Yields
    ------
    tuple[np.ndarray, np.ndarray]
        A tuple containing the bin edges and counts for the histogram after each chunk is processed.
    """
    n = min(max_samples, data.size)
    chunk_sizes = _chunk_sizes(data)
    rng = np.random.default_rng(0)
    n_chunks = len(chunk_sizes)
    n_selected = min(n_chunks, max(1, n // max(1, min(chunk_sizes))))
    probs = np.asarray(chunk_sizes) / sum(chunk_sizes)
    order = rng.choice(n_chunks, size=n_selected, p=probs, replace=False)

    range_min, range_max = contrast_limits_range

    running_counts = np.zeros(bin_count, dtype=np.float64)
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
            bins=bin_count,
            range=(float(range_min), float(range_max)),
        )
        running_counts += chunk_counts.astype(np.float64)

        bins_arr = np.linspace(range_min, range_max, bin_count + 1).astype(
            np.float32
        )
        if log_scale:
            counts = np.log10(running_counts + 1).astype(np.float32)
        else:
            counts = running_counts.astype(np.float32)

        yield bins_arr, counts


## CACHING


def _object_token(obj: Any) -> int | None:
    """Return a token identifying *obj* for as long as the object is alive."""
    if obj is None:
        return None
    key = id(obj)
    entry = _tokens.get(key)
    if entry is not None:
        token, ref = entry
        if ref() is obj:
            return token
    try:
        ref = weakref.ref(obj)
    except TypeError:
        # object cannot be weak referenced, fall back to its address
        return -key
    if len(_tokens) > _MAX_TOKENS:
        _prune_tokens()
    token = next(_token_counter)
    _tokens[key] = (token, ref)
    return token


def _prune_tokens() -> None:
    """Drop tokens of objects that are no longer alive."""
    for key in [k for k, (_token, ref) in _tokens.items() if ref() is None]:
        del _tokens[key]


def histogram_fingerprint(
    layer: Image | Surface, mode: Literal['canvas', 'full'] | None = None
) -> dict[str, Any] | None:
    """Compute the histogram fingerprint, uniquely identifying a computable histogram.

    Parameters
    ----------
    layer : Image | Surface
        The layer the histogram is computed for.
    mode : {'canvas', 'full'}, optional
        The computation mode. Defaults to the mode of the layer histogram
        model.

    Returns
    -------
    dict
        Dictionary of plain values describing the inputs of the computation.
        Two states are equal if and only if the same histogram would be
        computed for them.
    """
    try:
        with _lock:
            model = layer.histogram
            mode = model.mode if mode is None else mode
            source = (
                _canvas_source(layer)
                if mode == 'canvas'
                else _full_source(layer)
            )
            return {
                'mode': mode,
                'bins': int(model.bins),
                'max_samples': int(model.max_samples),
                'log_scale': bool(model.log_scale),
                'rgb': bool(getattr(layer, 'rgb', False)),
                'contrast_limits_range': (
                    float(layer.contrast_limits_range[0]),
                    float(layer.contrast_limits_range[1]),
                ),
                'source': _object_token(source),
            }
    except (AttributeError, IndexError, KeyError, TypeError, ValueError):
        # can happen if the layer is not yet sliced and we try to compute
        # the histogram
        return None


def get_cached_histogram(
    layer: Image | Surface,
    mode: Literal['canvas', 'full'] | None = None,
    fingerprint: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return the cached histogram of *layer* if it is still up to date.

    Parameters
    ----------
    layer : Image | Surface
        The layer to look up.
    mode : {'canvas', 'full'}, optional
        The computation mode. Defaults to the mode of the layer histogram
        model.
    fingerprint : dict, optional
        The fingerprint to compare against. Defaults to the current fingerprint of the
        layer.

    Returns
    -------
    tuple of (numpy.ndarray, numpy.ndarray) or None
        The cached ``bin_edges`` and ``counts``, or None if there is no
        up-to-date cached histogram.
    """
    with _lock:
        values = layer.metadata.get(VALUES_KEY)
        fingerprints = layer.metadata.get(FINGERPRINT_KEY)
        if not isinstance(values, dict) or not isinstance(fingerprints, dict):
            return None
        mode = layer.histogram.mode if mode is None else mode
        entry = values.get(mode)
        cached_fingerprint = fingerprints.get(mode)
        if not isinstance(entry, dict) or not isinstance(
            cached_fingerprint, dict
        ):
            return None
        if fingerprint is None:
            fingerprint = histogram_fingerprint(layer, mode)
            if fingerprint is None:
                return None
        if cached_fingerprint != fingerprint:
            return None
        bin_edges = entry.get('bin_edges')
        counts = entry.get('counts')
        if bin_edges is None or counts is None:
            return None
        return bin_edges, counts


def set_cached_histogram(
    layer: Image | Surface,
    bin_edges: np.ndarray,
    counts: np.ndarray,
    fingerprint: dict[str, Any] | None = None,
) -> None:
    """Store computed histogram values and the fingerprint they were computed for.

    Parameters
    ----------
    layer : Image | Surface
        The layer the histogram was computed for.
    bin_edges : numpy.ndarray
        The bin edges of the histogram.
    counts : numpy.ndarray
        The counts of the histogram.
    fingerprint : dict, optional
        The fingerprint the values were computed for. Defaults to the current fingerprint
        of the layer.
    """
    with _lock:
        if fingerprint is None:
            fingerprint = histogram_fingerprint(layer)
            if fingerprint is None:
                return
        mode = fingerprint['mode']
        values = layer.metadata.get(VALUES_KEY)
        if not isinstance(values, dict):
            values = {}
            layer.metadata[VALUES_KEY] = values
        fingerprints = layer.metadata.get(FINGERPRINT_KEY)
        if not isinstance(fingerprints, dict):
            fingerprints = {}
            layer.metadata[FINGERPRINT_KEY] = fingerprints
        values[mode] = {
            'bin_edges': np.array(bin_edges),
            'counts': np.array(counts),
        }
        fingerprints[mode] = dict(fingerprint)


def clear_cached_histogram(
    layer: Image | Surface, mode: Literal['canvas', 'full'] | None = None
) -> None:
    """Remove the cached histogram values of *layer*.

    Parameters
    ----------
    layer : Image | Surface
        The layer to clear the cache for.
    mode : {'canvas', 'full'}, optional
        The computation mode to clear. Defaults to clearing all modes.
    """
    with _lock:
        if mode is None:
            layer.metadata.pop(VALUES_KEY, None)
            layer.metadata.pop(FINGERPRINT_KEY, None)
            return
        values = layer.metadata.get(VALUES_KEY)
        if isinstance(values, dict):
            values.pop(mode, None)
        fingerprints = layer.metadata.get(FINGERPRINT_KEY)
        if isinstance(fingerprints, dict):
            fingerprints.pop(mode, None)
