from itertools import count

_request_ids = count()


def _next_request_id() -> int:
    """Returns the next ID associated with a slice."""
    return next(_request_ids)
