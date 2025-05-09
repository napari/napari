from napari.components import ViewerModel
from napari_builtins._widgets import FeaturesTable


def test_features_table(qtbot):
    v = ViewerModel()
    w = FeaturesTable(v)

    assert w.table.model().columnCount() == 1  # 0 is index
    assert w.table.model().rowCount() == 0

    v.add_points([0, 0], features={'a': [1]})
    assert w.table.model().columnCount() == 2
    assert w.table.model().rowCount() == 1
