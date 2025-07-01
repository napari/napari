from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from pandas.core.generic import pandas_dtype
from qtpy.QtCore import QItemSelection, QItemSelectionModel, Qt
from qtpy.QtGui import QGuiApplication
from qtpy.QtWidgets import (
    QAbstractItemDelegate,
    QComboBox,
    QFileDialog,
    QLineEdit,
)

from napari.components import ViewerModel
from napari_builtins._qt.features_table import FeaturesTable, PandasModel


@pytest.mark.usefixtures('qapp')
def test_pandas_model():
    df = pd.DataFrame({'a': [1, 2, 3, 4], 'b': [True, False, True, False]})
    view = PandasModel(df)
    assert view.rowCount() == 4
    assert view.columnCount() == 3  # 0 is index
    assert view.data(view.index(0, 1)) == '1'
    assert view.headerData(1, Qt.Orientation.Horizontal) == 'a'
    assert view.headerData(2, Qt.Orientation.Horizontal) == 'b'
    assert view.headerData(1, Qt.Orientation.Vertical) == 1


@pytest.mark.usefixtures('qapp')
def test_pandas_model_flags():
    df = pd.DataFrame({'a': [1, 2, 3, 4], 'b': [True, False, True, False]})
    view = PandasModel(df)
    assert view.flags(view.index(5, 5)) == Qt.ItemFlag.ItemIsEnabled
    assert view.flags(view.index(5, 5)) == Qt.ItemFlag.ItemIsEnabled

    view.editable = True
    assert view.flags(view.index(0, 1)) & Qt.ItemFlag.ItemIsEditable
    assert view.flags(view.index(0, 2)) & Qt.ItemFlag.ItemIsUserCheckable


def test_pandas_model_set_data(qtbot):
    df = pd.DataFrame({'a': [1, 2, 3, 4], 'b': [True, False, True, False]})
    view = PandasModel(df)
    assert not view.setData(view.index(5, 5), 1)
    assert not view.setData(view.index(0, 0), 1)  # cannot edit index
    assert view.setData(view.index(0, 1), 5)
    assert df.loc[0, 'a'] == 5
    assert not view.setData(view.index(0, 1), 7, Qt.ItemDataRole.UserRole)
    assert not view.setData(
        view.index(0, 1), 'aaa'
    )  # cannot set string to int column

    with qtbot.waitSignal(view.dataChanged):
        assert view.setData(view.index(0, 2), False)
    assert not df.loc[0, 'b']

    with qtbot.waitSignal(view.dataChanged):
        assert view.setData(
            view.index(0, 2),
            Qt.CheckState.Checked,
            Qt.ItemDataRole.CheckStateRole,
        )
    assert df.loc[0, 'b']


def test_pandas_model_set_data_categorical(qtbot):
    df = pd.DataFrame(
        {
            'a': pd.Series(['a', 'b', 'a', 'b'], dtype='category'),
            'b': [1, 2, 3, 4],
        }
    )
    view = PandasModel(df)

    with qtbot.waitSignal(view.dataChanged):
        assert view.setData(view.index(0, 1), 'b')
    assert df.loc[0, 'a'] == 'b'

    with qtbot.waitSignal(view.dataChanged):
        assert view.setData(view.index(0, 1), 'a')
    assert df.loc[0, 'a'] == 'a'

    assert not view.setData(view.index(0, 1), 'c')  # not in categories


def test_features_table(qtbot):
    v = ViewerModel()
    w = FeaturesTable(v)
    qtbot.add_widget(w)
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
    w.table.sortByColumn(1, Qt.SortOrder.AscendingOrder)
    for i in range(3):
        assert proxy.data(proxy.index(i, 1), Qt.ItemDataRole.EditRole) == str(
            i
        )
        assert layer.features['a'][i] == original_a[i]

    # sorting bools should work
    w.table.sortByColumn(2, Qt.SortOrder.AscendingOrder)
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

    # test selection (with sorted rows)
    layer.selected_data = {1, 2}

    selected_rows = {
        proxy.mapToSource(raw_index).row()
        for raw_index in w.table.selectionModel().selectedIndexes()
    }
    assert selected_rows == layer.selected_data

    # click on top left cell
    idx = proxy.mapFromSource(proxy.sourceModel().index(0, 0))

    # select whole rows
    first_cell = proxy.index(idx.row(), 0)
    last_cell = proxy.index(idx.row(), proxy.columnCount() - 1)
    selection = QItemSelection()
    selection.select(first_cell, last_cell)

    w.table.selectionModel().select(
        selection,
        QItemSelectionModel.SelectionFlag.ClearAndSelect
        | QItemSelectionModel.SelectionFlag.Rows,
    )

    assert layer.selected_data == {0}


def test_features_table_selection_labels(qtbot):
    v = ViewerModel()
    w = FeaturesTable(v)
    qtbot.add_widget(w)

    original_a = (2, 0, 1)

    layer = v.add_labels(
        np.zeros((10, 10), dtype=np.uint8), features={'a': original_a}
    )
    assert layer.selected_label == 1

    w.table.selectRow(2)
    assert layer.selected_label == 2


def test_features_table_selection_shapes(qtbot):
    v = ViewerModel()
    features = pd.DataFrame(
        {'shape_type': ['rectangle', 'ellipse'], 'value': [1, 2]}
    )
    data = [
        np.array([[0, 0], [1, 1], [1, 0], [0, 1]]),
        np.array([[2, 2], [3, 3], [3, 2], [2, 3]]),
    ]
    layer = v.add_shapes(data, features=features)

    w = FeaturesTable(v)
    qtbot.add_widget(w)

    assert layer.selected_data == set()

    w.table.selectRow(1)
    assert layer.selected_data == {1}


def test_features_table_edit(qtbot):
    v = ViewerModel()
    w = FeaturesTable(v)
    qtbot.add_widget(w)
    proxy = w.table.model()

    original_a = ['x', 'y']

    layer = v.add_points(np.zeros((2, 2)), features={'a': original_a})

    idx = proxy.index(0, 1)
    w.table.edit(idx)
    assert not w.table.isPersistentEditorOpen(idx)

    w.toggle.click()
    assert proxy.sourceModel().editable
    w.table.edit(idx)
    assert w.table.isPersistentEditorOpen(idx)

    editor = w.table.findChild(QLineEdit)
    qtbot.keyClicks(editor, 'hello')
    assert editor.text() == 'hello'
    w.table.commitData(editor)
    assert layer.features.loc[0, 'a'] == 'hello'


def test_features_table_save_csv(qtbot, tmp_path, monkeypatch):
    v = ViewerModel()
    w = FeaturesTable(v)
    qtbot.add_widget(w)

    df = pd.DataFrame({'a': [1, 2]})
    v.add_points(np.zeros((2, 2)), features=df)

    path = tmp_path / 'test.csv'
    monkeypatch.setattr(
        QFileDialog, 'getSaveFileName', MagicMock(return_value=(path, None))
    )

    w.save.click()

    pd.testing.assert_frame_equal(pd.read_csv(path, index_col=0), df)


def test_features_table_copy_paste(qtbot, qapp):
    v = ViewerModel()
    w = FeaturesTable(v)
    qtbot.add_widget(w)
    proxy = w.table.model()

    df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})

    layer = v.add_points(np.zeros((3, 2)), features=df.copy())

    first_cell = proxy.index(1, 1)
    last_cell = proxy.index(1, 2)
    selection = QItemSelection()
    selection.select(first_cell, last_cell)

    w.table.selectionModel().select(
        selection,
        QItemSelectionModel.SelectionFlag.ClearAndSelect,
    )
    qapp.clipboard().setText('')
    qtbot.keyClick(w.table, 'c')
    assert qapp.clipboard().text().strip() == ''
    qtbot.keyClick(w.table, 'c', Qt.KeyboardModifier.ControlModifier)

    # stip cause windows and linux otherwise differ
    assert qapp.clipboard().text().strip() == '2\t5'

    first_cell = proxy.index(2, 1)
    last_cell = proxy.index(2, 2)
    selection = QItemSelection()
    selection.select(first_cell, last_cell)

    w.table.selectionModel().select(
        selection,
        QItemSelectionModel.SelectionFlag.ClearAndSelect,
    )

    w.toggle.click()
    QGuiApplication.clipboard().setText('3\t8\t7')
    # we test here that presence of additional columns does not
    # cause issues when pasting and we just discard them
    qtbot.keyClick(w.table, 'v', Qt.KeyboardModifier.ControlModifier)

    np.testing.assert_array_equal(layer.features.iloc[2], [3, 8])


@pytest.mark.parametrize(
    ('dtype', 'val', 'rendered_val', 'editor_class', 'new_val'),
    [
        (int, 2, '2', QLineEdit, 3),
        (float, 123.45678, '123.457', QLineEdit, 1e10),
        (
            'datetime64[ns]',
            '22-07-2025',
            '2025-07-22',
            QLineEdit,
            '2025-03-14',
        ),
        (bool, False, '', None, None),  # bool uses checkboxes instead
        (bool, True, '', None, None),  # bool uses checkboxes instead
        (pd.CategoricalDtype(['x', 'y']), 'x', 'x', QComboBox, 'y'),
    ],
)
def test_features_tables_dtypes(
    dtype, val, rendered_val, editor_class, new_val, qtbot
):
    v = ViewerModel()
    w = FeaturesTable(v)
    qtbot.add_widget(w)
    proxy = w.table.model()

    df = pd.DataFrame({'a': pd.Series([val], dtype=dtype)})

    layer = v.add_points(np.zeros((1, 2)), features=df)
    idx = proxy.index(
        0,
        1,
    )
    assert layer.features['a'].dtype == pandas_dtype(dtype)
    assert (
        proxy.data(
            idx,
            Qt.ItemDataRole.DisplayRole,
        )
        == rendered_val
    )

    w.toggle.click()
    w.table.edit(idx)

    if editor_class is None:
        # bools use checkboxes and not an editor, skip
        return

    editor = w.table.findChild(editor_class)

    if issubclass(editor_class, QLineEdit):
        qtbot.keyClicks(editor, str(new_val))
        w.table.commitData(editor)
    elif editor_class == QComboBox:
        qtbot.keyClick(editor, Qt.Key.Key_Down)
        w.table.commitData(editor)
    qtbot.wait(5)  # wait on singleShot execution
    w.table.closeEditor(editor, QAbstractItemDelegate.EndEditHint.NoHint)

    assert (
        layer.features.loc[0, 'a']
        == pd.Series(new_val, dtype=pandas_dtype(dtype))[0]
    )


def test_features_table_change_active_layer(qtbot):
    v = ViewerModel()
    w = FeaturesTable(v)
    qtbot.add_widget(w)

    df = pd.DataFrame({'a': [1, 2, 3]})
    layer1 = v.add_points(np.zeros((3, 2)), features=df.copy())
    layer2 = v.add_labels(
        np.zeros((10, 10), dtype=np.uint8), features=df.copy()
    )
    layer3 = v.add_image(np.empty((10, 10), dtype=np.uint8))

    v.layers.selection.active = layer1
    assert len(layer1.events.features.callbacks) == 2
    assert len(layer1.selected_data.events.all) == 1
    assert len(layer2.events.features.callbacks) == 1
    assert len(layer2.events.selected_label.callbacks) == 1
    assert w.info.text() == f'Features of "{layer1.name}"'

    v.layers.selection.active = layer2
    assert len(layer1.events.features.callbacks) == 1
    assert len(layer1.selected_data.events.all) == 0
    assert len(layer2.events.features.callbacks) == 2
    assert len(layer2.events.selected_label.callbacks) == 2
    assert w.info.text() == f'Features of "{layer2.name}"'

    v.layers.selection.active = layer3
    assert len(layer1.events.features.callbacks) == 1
    assert len(layer1.selected_data.events.all) == 0
    assert len(layer2.events.features.callbacks) == 1
    assert len(layer2.events.selected_label.callbacks) == 1
    assert 'has no features table' in w.info.text()

    v.layers.selection.active = None
    assert len(layer1.events.features.callbacks) == 1
    assert len(layer2.events.features.callbacks) == 1
    assert 'No layer selected.' in w.info.text()
