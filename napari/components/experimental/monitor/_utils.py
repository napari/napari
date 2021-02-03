"""Monitor Utilities.
"""
import base64
import json

import numpy as np


class NumpyJSONEncoder(json.JSONEncoder):
    """A JSONEncoder that also converts ndarray's to lists."""

    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        return json.JSONEncoder.default(self, o)


def numpy_dumps(data: dict) -> str:
    """Return data as a JSON string.

    Returns
    -------
    str
        The JSON string.
    """
    return json.dumps(data, cls=NumpyJSONEncoder)


def base64_encoded_json(data: dict) -> str:
    """Return base64 encoded version of this data as JSON.

    Parameters
    ----------
    data : dict
        The data to write as JSON then base64 encode.

    Returns
    -------
    str
        The base64 encoded JSON string.
    """
    json_str = numpy_dumps(data)
    json_bytes = json_str.encode('ascii')
    message_bytes = base64.b64encode(json_bytes)
    return message_bytes.decode('ascii')
