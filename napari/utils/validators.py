from collections.abc import Collection


def validate_N_seq(n: int, dtype=None):
    """Creates a function to validate a sequence of len == N and type == dtype.

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
    """

    def func(obj):
        if not (isinstance(obj, Collection) and hasattr(obj, '__getitem__')):
            raise TypeError(
                "object must be an indexable collection "
                f"(list, tuple, or np.array), of length {n}"
            )
        if not len(obj) == n:
            raise ValueError(f"object must have length {n}, got {len(obj)}")
        if dtype is not None:
            for item in obj:
                if not isinstance(item, dtype):
                    raise TypeError(
                        f"Every item in the sequence must be of type {dtype}, "
                        f"but {item} is of type {type(item)}"
                    )

    return func
