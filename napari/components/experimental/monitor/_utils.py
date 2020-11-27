"""Monitor Utilities.
"""
import json

import numpy as np


class NumpyJSONEncoder(json.JSONEncoder):
    """A JSONEncoder that also converts ndarray's to lists."""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def numpy_dumps(data: dict) -> str:
    """Return data as a JSON string.

    Return
    ------
    str
        The JSON string.
    """
    return json.dumps(data, cls=NumpyJSONEncoder)
