import numpy as np
import numpy.typing as npt


def elements_in_slice(
    index: tuple[npt.NDArray[np.int_], ...], position_in_axes: dict[int, int]
) -> npt.NDArray[np.bool_]:
    """Mask elements from a multi-dimensional index not in a given slice.

    Some n-D operations may edit data that is not visible in the current slice.
    Given slice position information (as a dictionary mapping axis to index on that
    axis), this function returns a boolean mask for the possibly higher-dimensional
    multi-index so that elements not currently visible are masked out. The
    resulting multi-index can then be subset and used to index into a texture or
    other lower-dimensional view.

    Parameters
    ----------
    index : tuple of array of int
        A NumPy fancy indexing expression [1]_.
    position_in_axes : dict[int, int]
        A dictionary mapping sliced (non-displayed) axes to a slice position.

    Returns
    -------
    visible : array of bool
        A boolean array indicating which items are visible in the current view.
    """
    queries = [
        index[ax] == position for ax, position in position_in_axes.items()
    ]
    return np.logical_and.reduce(queries, axis=0)


def index_in_slice(
    index: tuple[npt.NDArray[np.int_], ...],
    position_in_axes: dict[int, int],
    indices_order: tuple[int, ...],
) -> tuple[npt.NDArray[np.int_], ...]:
    """Convert a NumPy fancy indexing expression from data to sliced space.

    Parameters
    ----------
    index : tuple of array of int
        A NumPy fancy indexing expression [1]_.
    position_in_axes : dict[int, int]
        A dictionary mapping sliced (non-displayed) axes to a slice position.
    indices_order : tuple of int
        The order of the indices in data view.

    Returns
    -------
    sliced_index : tuple of array of int
        The indexing expression (nD) restricted to the current slice (usually
        2D or 3D).

    Examples
    --------
    >>> index = (np.arange(5), np.full(5, 1), np.arange(4, 9))
    >>> index_in_slice(index, {0: 3}, (0, 1, 2))
    (array([1]), array([7]))
    >>> index_in_slice(index, {1: 1, 2: 8}, (0, 1, 2))
    (array([4]),)

    References
    ----------
    [1]: https://numpy.org/doc/stable/user/basics.indexing.html#integer-array-indexing
    """
    index_in_slice = elements_in_slice(index, position_in_axes)
    return tuple(
        index[i][index_in_slice]
        for i in indices_order
        if i not in position_in_axes
    )
