import numpy as np
from qtpy.QtCore import Qt

from napari.components import ViewerModel
from napari_builtins._qt.features_table import FeaturesTable


def _detect_warn(*args):
    if args[-1] == 'edit: editing failed':
        raise RuntimeError
    if args[-1] == 'edit: index was invalid':
        raise KeyError


def test_features_table(qtbot):
    v = ViewerModel()
    w = FeaturesTable(v)
    proxy = w.table.model()

    assert proxy.columnCount() == 1  # 0 is index
    assert proxy.rowCount() == 0

    original_a = (2, 0, 1)

    layer = v.add_points(np.zeros((3, 2)), features={'a': original_a})
    assert proxy.columnCount() == 2
    assert proxy.rowCount() == 3

    layer.features['b'] = [True, False, True]
    layer.events.features()

    assert proxy.columnCount() == 3
    assert proxy.rowCount() == 3

    # sorting should sort the proxy model but not the data
    w.table.sortByColumn(1, Qt.AscendingOrder)
    for i in range(3):
        assert proxy.data(proxy.index(i, 1), Qt.ItemDataRole.EditRole) == str(
            i
        )
        assert layer.features['a'][i] == original_a[i]

    # sorting bools should work
    w.table.sortByColumn(2, Qt.AscendingOrder)
    for i in range(3):
        assert (
            proxy.data(
                proxy.index(
                    i,
                    2,
                ),
                Qt.ItemDataRole.EditRole,
            )
            == sorted(layer.features['b'])[i]
        )
