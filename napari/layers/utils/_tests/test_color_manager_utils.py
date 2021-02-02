import numpy as np

from napari.layers.utils.color_manager_utils import guess_continuous


def test_guess_continuous():
    continuous_annotation = np.array([1, 2, 3], dtype=np.float32)
    assert guess_continuous(continuous_annotation)

    categorical_annotation_1 = np.array([True, False], dtype=bool)
    assert not guess_continuous(categorical_annotation_1)

    categorical_annotation_2 = np.array([1, 2, 3], dtype=np.int)
    assert not guess_continuous(categorical_annotation_2)
