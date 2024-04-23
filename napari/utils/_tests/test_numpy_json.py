import numpy as np

from napari.utils._numpy_json import NumpyEncoder


def test_numpy_encoder():
    data = {'a': np.array([1, 2, 3]), 'b': np.array([[1, 2], [3, 4]]), 'c': 1}
    encoder = NumpyEncoder()
    assert (
        encoder.encode(data)
        == '{"a": [1, 2, 3], "b": [[1, 2], [3, 4]], "c": 1}'
    )
