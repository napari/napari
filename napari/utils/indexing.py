from typing import Dict, Tuple

import numpy as np
import numpy.typing as npt


def index_in_slice(
    index: Tuple[npt.NDArray[np.int_], ...], position_in_axes: Dict[int, int]
) -> Tuple[npt.NDArray[np.int_], ...]:
    """Convert a NumPy fancy indexing expression from data to sliced space.

    Parameters
    ----------
    index : tuple of array of int
        A NumPy fancy indexing expression [1]_.
    position_in_axes : dict[int, int]
        A dictionary mapping sliced (non-displayed) axes to a slice position.

    Returns
    -------
    sliced_index : tuple of array of int
        The indexing expression (nD) restricted to the current slice (usually
        2D or 3D).

    Examples
    --------
    >>> index = (np.arange(5), np.full(5, 1), np.arange(4, 9))
    >>> index_in_slice(index, {0: 3})
    (array([1]), array([7]))
    >>> index_in_slice(index, {1: 1, 2: 8})
    (array([4]),)

    References
    ----------
    [1]: https://numpy.org/doc/stable/user/basics.indexing.html#integer-array-indexing
    """
    queries = [
        index[ax] == position for ax, position in position_in_axes.items()
    ]
    index_in_slice = np.logical_and.reduce(queries, axis=0)
    return tuple(
        ix[index_in_slice]
        for i, ix in enumerate(index)
        if i not in position_in_axes
    )
