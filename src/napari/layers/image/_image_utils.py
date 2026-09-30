"""guess_rgb, guess_multiscale, guess_labels."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from napari.layers._data_protocols import LayerDataProtocol, assert_protocol
from napari.layers.multiscale_data import MultiScaleData


def guess_rgb(shape: tuple[int, ...], min_side_len: int = 30) -> bool:
    """Guess if the passed shape comes from rgb data.

    If last dim is 3 or 4 and other dims are larger (>30), assume the data is
    rgb, including rgba.

    Parameters
    ----------
    shape : list of int
        Shape of the data that should be checked.

    Returns
    -------
    bool
        If data is rgb or not.
    """
    ndim = len(shape)
    last_dim = shape[-1]
    viewed_dims = shape[-3:-1]

    return (
        ndim > 2
        and last_dim in (3, 4)
        and all(d > min_side_len for d in viewed_dims)
    )


def guess_multiscale(
    data: MultiScaleData | list | tuple | LayerDataProtocol,
) -> tuple[bool, LayerDataProtocol | MultiScaleData]:
    """Guess whether the passed data is multiscale, process it accordingly.

    If `data` is already a single array-like with more than one dimension,
    it is not multiscale. If `data` is a sequence containing a single
    element, that element is unwrapped and treated as non-multiscale data.
    Otherwise, `data` is assumed to be a sequence of multiscale levels and
    is validated as such (see :func:`validate_multiscale_data`): levels
    must all have the same number of dimensions and non-increasing size,
    or a ValueError is raised.

    Parameters
    ----------
    data : array or list of array
        Data that should be checked.

    Returns
    -------
    multiscale : bool
        True if the data is thought to be multiscale, False otherwise.
    data : list or array
        The input data, perhaps unwrapped if it contained a single element.

    Raises
    ------
    ValueError
        If `data` is a sequence of more than one array-like whose levels
        are not non-increasing in size, or do not all have the same
        number of dimensions.
    TypeError
        If any item in `data` does not implement `LayerDataProtocol`.
    """
    if isinstance(data, MultiScaleData):
        return True, data

    try:
        assert_protocol(data)
        # 1D array-likes cannot be scalar layer data, must be treated as
        # a candidate sequence of multiscale levels
        is_multiscale = data.ndim == 1
        multiscale_len = data.shape[0]
    except TypeError:
        # not itself layer data, has to be a sequence of levels
        is_multiscale = True
        multiscale_len = len(data)

    if not is_multiscale:
        return False, data

    if multiscale_len == 1:
        # pyramid with only one level, unwrap
        return False, data[0]

    return True, MultiScaleData(data)


def guess_labels(data: Any) -> Literal['labels', 'image']:
    """Guess if array contains labels data."""

    if hasattr(data, 'dtype') and data.dtype in (
        np.int32,
        np.uint32,
        np.int64,
        np.uint64,
    ):
        return 'labels'

    return 'image'
