from __future__ import annotations

from itertools import tee
from typing import TYPE_CHECKING, TypeGuard, TypeVar

from napari.utils.translations import trans

if TYPE_CHECKING:
    from collections.abc import MutableSequence

T = TypeVar('T', bound=int | float)


# TODO: this should be generalizable to
# other mutable sequence types (e.g., tuple)
def check_list(
    values: list[T | None], n: int
) -> TypeGuard[list[T]]:
    """
    Check that all elements in the list are not None,
    and that the length of the iterable is n.

    Returns
    -------
    bool
        True if all elements are not None and length is n, False otherwise.
    """
    return all(item is not None for item in values) and len(values) == n


def _pairwise(iterable: MutableSequence[T]) -> zip[tuple[T, T]]:
    """Convert iterable to a zip object containing tuples of pairs along the
    sequence.

    Examples
    --------
    >>> pairwise([1, 2, 3, 4])
    <zip at 0x10606df80>

    >>> list(pairwise([1, 2, 3, 4]))
    [(1, 2), (2, 3), (3, 4)]
    """
    # duplicate the iterable
    a, b = tee(iterable)
    # shift b by one position
    next(b, None)
    # create tuple pairs from the values in a and b
    return zip(a, b, strict=False)


def validate_increasing(values: list[T]) -> None:
    """Ensure that values in an iterable are monotocially increasing.

    Examples
    --------
    >>> validate_increasing([1, 2, 3, 4])
    None

    >>> validate_increasing([1, 4, 3, 4])
    ValueError: Sequence [1, 4, 3, 4] must be monotonically increasing.

    Raises
    ------
    ValueError
        If `values` is constant or decreasing from one value to the next.
    """
    # convert iterable to pairwise tuples, check each tuple
    if any(a >= b for a, b in _pairwise(values)):
        raise ValueError(
            trans._(
                'Sequence {sequence} must be monotonically increasing.',
                deferred=True,
                sequence=values,
            )
        )
