from collections.abc import Collection, Generator

from .translations import trans


def validate_n_seq(n: int, dtype=None):
    """Creates a function to validate a sequence of len == N and type == dtype.

    Currently does **not** validate generators (will always validate true).

    Parameters
    ----------
    n : int
        Desired length of the sequence
    dtype : type, optional
        If provided each item in the sequence must match dtype, by default None

    Returns
    -------
    function
        Function that can be called on an object to validate that is a sequence
        of len `n` and (optionally) each item in the sequence has type `dtype`

    Examples
    --------
    >>> validate = validate_N_seq(2)
    >>> validate(8)  # raises TypeError
    >>> validate([1, 2, 3])  # raises ValueError
    >>> validate([4, 5])  # just fine, thank you very much
    """

    def func(obj):
        """Function that validates whether an object is a sequence of len `n`.

        Parameters
        ----------
        obj : any
            the object to be validated

        Raises
        ------
        TypeError
            If the object is not an indexable collection.
        ValueError
            If the object does not have length `n`
        TypeError
            If `dtype` was provided to the wrapper function and all items in
            the sequence are not of type `dtype`.
        """

        if isinstance(obj, Generator):
            return
        if not (isinstance(obj, Collection) and hasattr(obj, '__getitem__')):
            raise TypeError(
                trans._(
                    "object '{obj}' is not an indexable collection (list, tuple, or np.array), of length {number}",
                    deferred=True,
                    obj=obj,
                    number=n,
                )
            )
        if len(obj) != n:
            raise ValueError(
                trans._(
                    "object must have length {number}, got {obj_len}",
                    deferred=True,
                    number=n,
                    obj_len=len(obj),
                )
            )
        if dtype is not None:
            for item in obj:
                if not isinstance(item, dtype):
                    raise TypeError(
                        trans._(
                            "Every item in the sequence must be of type {dtype}, but {item} is of type {item_type}",
                            deferred=True,
                            dtype=dtype,
                            item=item,
                            item_type=type(item),
                        )
                    )

    return func
