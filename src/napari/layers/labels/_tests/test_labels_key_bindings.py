from unittest.mock import Mock

import numpy as np
import pytest

from napari.layers import Labels
from napari.layers.labels import _labels_key_bindings
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


def test_guard_for_out_of_range_selected_label(monkeypatch):
    show_warning_mock = Mock()
    monkeypatch.setattr(
        _labels_key_bindings, 'show_warning', show_warning_mock
    )
    labels = Labels(np.zeros((10, 10), dtype=np.uint8))

    labels.selected_label = 0
    decrease_label_id(labels)
    assert labels.selected_label == 0
    show_warning_mock.assert_called_once()
    show_warning_mock.call_args_list[0][0][0].startswith('The value -1')
    show_warning_mock.reset_mock()

    labels.selected_label = 255
    increase_label_id(labels)
    assert labels.selected_label == 255
    show_warning_mock.assert_called_once()
    show_warning_mock.call_args_list[0][0][0].startswith('The value 256')


def test_label_overflow(monkeypatch):
    show_warning_mock = Mock()
    monkeypatch.setattr(
        _labels_key_bindings, 'show_warning', show_warning_mock
    )
    data = np.zeros((10, 10), dtype=np.uint8)
    labels = Labels(data)
    new_label(labels)
    assert labels.selected_label == 1
    show_warning_mock.assert_not_called()

    data[0, 0] = 255
    new_label(labels)
    assert labels.selected_label == 1
    show_warning_mock.assert_called_once()
    show_warning_mock.call_args_list[0][0][0].startswith('The value 256')
