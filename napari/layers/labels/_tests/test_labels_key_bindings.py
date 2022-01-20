from tempfile import TemporaryDirectory

import numpy as np
import pytest
import zarr

from napari.layers import Labels
from napari.layers.labels._labels_key_bindings import new_label


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


def test_max_label_tensorstore(labels_data_4d):
    ts = pytest.importorskip('tensorstore')

    with TemporaryDirectory(suffix='.zarr') as fout:
        labels_temp = zarr.open(
            fout,
            mode='w',
            shape=labels_data_4d.shape,
            dtype=np.uint32,
            chunks=(1, 1, 8, 9),
        )
        labels_temp[:] = labels_data_4d
        labels_ts_spec = {
            'driver': 'zarr',
            'kvstore': {'driver': 'file', 'path': fout},
            'path': '',
            'metadata': {
                'dtype': labels_temp.dtype.str,
                'order': labels_temp.order,
                'shape': labels_data_4d.shape,
            },
        }
        data = ts.open(labels_ts_spec, create=False, open=True).result()
        layer = Labels(data)
        new_label(layer)
        assert layer.selected_label == 4
