"""Pure utility functions for histogram computation.

All functions in this module are free of layer, Qt, and vispy dependencies.
They operate only on numpy arrays and numeric parameters, making them
suitable for use in any context (model, widget, or test).
"""

from __future__ import annotations

import numpy as np


def compute_histogram(
    data: np.ndarray,
    n_bins: int = 256,
    range_: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute histogram bin edges and counts for any dtype.

    Uses np.bincount for uint8 data (up to 256 values) and
    np.histogram for all other types.

    Parameters
    ----------
    data : np.ndarray
        Input data array. Will be raveled before computation.
    n_bins : int, default: 256
        Number of histogram bins.
    range_ : tuple[float, float], optional
        (min, max) of the histogram range. If None, inferred from data.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (bin_edges, counts) — bin edges as float32, counts as float32.
    """
    if data.size == 0:
        return np.array([0.0, 1.0], dtype=np.float32), np.array(
            [0.0], dtype=np.float32
        )

    data = np.asarray(data).ravel()
    # Filter out non-finite values
    valid = np.isfinite(data)
    if not np.any(valid):
        return np.array([0.0, 1.0], dtype=np.float32), np.array(
            [0.0], dtype=np.float32
        )
    data = data[valid]

    if range_ is None:
        range_ = (float(np.nanmin(data)), float(np.nanmax(data)))

    range_min, range_max = range_
    if range_min == range_max:
        range_min -= 0.5
        range_max += 0.5

    # Use np.bincount for small-integer data
    dtype = data.dtype
    if (
        np.issubdtype(dtype, np.unsignedinteger)
        and np.iinfo(dtype).max < 65536
    ):
        # For uint8/uint16, use bincount for speed
        counts = np.bincount(data.astype(np.int64), minlength=0).astype(
            np.float32
        )
        # If n_bins is large enough to cover the range, return bincount result
        if len(counts) <= n_bins:
            # Pad or truncate to n_bins
            if len(counts) < n_bins:
                counts = np.pad(counts, (0, n_bins - len(counts)))
            bin_edges = np.linspace(
                range_min, range_max, n_bins + 1, dtype=np.float32
            )
            return bin_edges, counts[:n_bins]

    # Fall back to np.histogram for floats, signed ints, or large uint ranges
    counts, bins = np.histogram(
        data,
        bins=n_bins,
        range=(float(range_min), float(range_max)),
    )
    return bins.astype(np.float32), counts.astype(np.float32)


def log_transform(counts: np.ndarray, base: float = 10.0) -> np.ndarray:
    """Apply log transform to histogram counts.

    Uses log(counts + 1) to avoid -inf from log(0).

    Parameters
    ----------
    counts : np.ndarray
        Histogram counts to transform.
    base : float, default: 10.0
        Logarithm base (e.g., 10 for log10, e for natural log).

    Returns
    -------
    np.ndarray
        Log-transformed counts as float32.
    """
    if base == 10.0:
        return np.log10(counts + 1).astype(np.float32)
    return (np.log(counts + 1) / np.log(base)).astype(np.float32)


def auto_bins(
    data: np.ndarray | np.dtype,
    max_bins: int = 256,
) -> int:
    """Guess optimal bin count from data dtype and range.

    Uses the full value range for small unsigned integer types
    (e.g., 256 bins for uint8) and caps at max_bins for all others.

    Parameters
    ----------
    data : np.ndarray or np.dtype
        Data array or dtype to compute bin count for.
    max_bins : int, default: 256
        Maximum number of bins to return.

    Returns
    -------
    int
        Recommended number of bins.
    """
    dtype = np.dtype(data) if isinstance(data, np.dtype) else data.dtype
    if np.issubdtype(dtype, np.unsignedinteger):
        info = np.iinfo(dtype)
        return min(int(info.max) + 1, max_bins)
    return max_bins


def downsample_histogram(
    counts: np.ndarray,
    edges: np.ndarray | None = None,
    max_bins: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce bin count by averaging adjacent bins.

    When the number of bins exceeds max_bins, this function groups
    adjacent bins and averages their counts, producing a smaller
    histogram that preserves the shape of the distribution.

    Parameters
    ----------
    counts : np.ndarray
        Original histogram counts (1D).
    edges : np.ndarray, optional
        Original bin edges (1D, length len(counts) + 1).
        If provided, bin edges are also downsampled.
    max_bins : int, default: 256
        Maximum number of bins in the output.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (downsampled_counts, downsampled_edges).
    """
    n_bins = len(counts)
    if n_bins <= max_bins:
        if edges is not None:
            return counts, edges
        return counts, edges  # type: ignore[return-value]

    # Calculate the grouping factor
    group_size = int(np.ceil(n_bins / max_bins))
    new_n_bins = n_bins // group_size

    # Truncate to multiple of group_size for clean reshape
    trimmed = counts[: new_n_bins * group_size]
    downsampled = trimmed.reshape(new_n_bins, group_size).mean(axis=1)

    if edges is not None:
        # Sample edges at the start of each group
        downsampled_edges = edges[::group_size][: new_n_bins + 1]
        # Ensure we have the right number of edges
        if len(downsampled_edges) < new_n_bins + 1:
            downsampled_edges = np.append(downsampled_edges, edges[-1:])
        return downsampled.astype(np.float32), downsampled_edges.astype(
            np.float32
        )

    return downsampled.astype(np.float32), edges  # type: ignore[return-value]


def crop_to_range(
    counts: np.ndarray,
    edges: np.ndarray,
    range_: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Crop histogram to a visible data range.

    Parameters
    ----------
    counts : np.ndarray
        Histogram counts (1D).
    edges : np.ndarray
        Bin edges (1D, length len(counts) + 1).
    range_ : tuple[float, float]
        (min, max) visible range to crop to.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (cropped_counts, cropped_edges).
    """
    range_min, range_max = range_
    if range_min >= range_max:
        return counts, edges

    # Find the bin indices corresponding to the range
    left_idx = np.searchsorted(edges, range_min, side='right') - 1
    left_idx = max(0, left_idx)

    right_idx = np.searchsorted(edges, range_max, side='left')
    right_idx = min(len(edges) - 1, right_idx)

    if right_idx <= left_idx:
        # Range is narrower than a single bin; return the bin that contains it
        bin_idx = min(left_idx, len(counts) - 1)
        return counts[bin_idx : bin_idx + 1], edges[bin_idx : bin_idx + 2]

    return counts[left_idx:right_idx], edges[left_idx : right_idx + 1]
