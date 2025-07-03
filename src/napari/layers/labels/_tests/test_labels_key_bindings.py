import numpy as np
import pytest

from napari.layers import Labels
from napari.layers.labels._labels_key_bindings import (
    decrease_label_id,
    increase_label_id,
    new_label,
    swap_selected_and_background_labels,
)


@pytest.fixture
def labels_data_4d():
    labels = np.zeros((5, 7, 8, 9), dtype=int)
    labels[1, 2:4, 4:6, 4:6] = 1
    labels[1, 3:5, 5:7, 6:8] = 2
    labels[2, 3:5, 5:7, 6:8] = 3
    return labels


def test_max_label(labels_data_4d):
    labels = Labels(labels_data_4d)
    new_label(labels)
    assert labels.selected_label == 4


def test_swap_background_label(labels_data_4d):
    labels = Labels(labels_data_4d)
    labels.selected_label = 10
    swap_selected_and_background_labels(labels)
    assert labels.selected_label == labels.colormap.background_value
    swap_selected_and_background_labels(labels)
    assert labels.selected_label == 10


def test_predefined_labels_switching(labels_data_4d):
    predefined_labels = [21, 1, 20, 30, 40, 2, 10]
    labels = Labels(labels_data_4d, predefined_labels=predefined_labels)
    predefined_labels = sorted(predefined_labels)

    labels.selected_label = 1
    for label_id in predefined_labels[1:]:
        increase_label_id(labels)
        assert labels.selected_label == label_id

    for _i in range(3):
        increase_label_id(labels)
        assert labels.selected_label == predefined_labels[-1]

    for label_id in predefined_labels[::-1][1:]:
        decrease_label_id(labels)
        assert labels.selected_label == label_id

    for _i in range(3):
        decrease_label_id(labels)
        assert labels.selected_label == min(
            predefined_labels[0], labels.colormap.background_value
        )
