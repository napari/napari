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
from napari_builtins._qt.features_table import FeaturesTable


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


def test_features_table_edit(qtbot):
    v = ViewerModel()
    w = FeaturesTable(v)
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


def test_features_table_save_csv(qtbot, tmp_path):
    v = ViewerModel()
    w = FeaturesTable(v)

    df = pd.DataFrame({'a': [1, 2]})
    v.add_points(np.zeros((2, 2)), features=df)

    path = tmp_path / 'test.csv'
    QFileDialog.getSaveFileName = MagicMock(return_value=(path, None))

    w.save.click()

    pd.testing.assert_frame_equal(pd.read_csv(path, index_col=0), df)


def test_features_table_copy_paste(qtbot):
    v = ViewerModel()
    w = FeaturesTable(v)
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

    w.table.copySelection()

    # stip cause windows and linux otherwise differ
    assert QGuiApplication.clipboard().text().strip() == '2\t5'

    first_cell = proxy.index(2, 1)
    last_cell = proxy.index(2, 2)
    selection = QItemSelection()
    selection.select(first_cell, last_cell)

    w.table.selectionModel().select(
        selection,
        QItemSelectionModel.SelectionFlag.ClearAndSelect,
    )

    w.toggle.click()
    QGuiApplication.clipboard().setText('2\t5')
    w.table.pasteSelection()

    np.testing.assert_array_equal(layer.features.iloc[2], df.iloc[1])


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
