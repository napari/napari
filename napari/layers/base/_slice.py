from itertools import count

# We use an incrementing non-negative integer to uniquely identify
# slices that is unbounded based on Python 3's int.
_request_ids = count()


def _next_request_id() -> int:
    """Returns the next integer identifier associated with a slice."""
    return next(_request_ids)
