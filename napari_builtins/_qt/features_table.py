from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from qtpy.QtCore import (
    QAbstractTableModel,
    QItemSelection,
    QItemSelectionModel,
    QModelIndex,
    QSortFilterProxyModel,
    Qt,
    QTimer,
)
from qtpy.QtGui import (
    QGuiApplication,
    QKeySequence,
)
from qtpy.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFileDialog,
    QLabel,
    QPushButton,
    QStyledItemDelegate,
    QTableView,
    QVBoxLayout,
    QWidget,
)
from superqt import QToggleSwitch

from napari.utils.history import get_save_history
from napari.utils.misc import in_ipython

if TYPE_CHECKING:
    import napari


class PandasModel(QAbstractTableModel):
    """Qt Model for a pandas DataFrame."""

    def __init__(self, df=None, parent=None):
        super().__init__(parent)
        self.df = df if df is not None else pd.DataFrame()
        self.editable = False

    # model methods necessary for qt
    def rowCount(self, parent=None):
        return self.df.shape[0]

    def columnCount(self, parent=None):
        return self.df.shape[1] + 1  # include index

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None

        row = index.row()
        col = index.column()

        if col == 0:  # index
            if role in {Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.EditRole}:
                return str(self.df.index[row])
            return None

        value = self.df.iat[row, col - 1]
        dtype = self.df.dtypes.iat[col - 1]

        # show booleans as respective checkboxes
        if role == Qt.CheckStateRole and pd.api.types.is_bool_dtype(dtype):
            return Qt.Checked if value else Qt.Unchecked

        if role in {Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.EditRole}:
            # format based on dtype
            if pd.api.types.is_float_dtype(dtype):
                return f'{value:.6g}'
            if pd.api.types.is_integer_dtype(dtype):
                return f'{value:d}'
            if pd.api.types.is_datetime64_any_dtype(dtype):
                return value.strftime('%Y-%m-%d')
            if pd.api.types.is_bool_dtype(dtype):
                if role == Qt.ItemDataRole.DisplayRole:
                    return ''  # do not show True/False text
                if role == Qt.ItemDataRole.EditRole:
                    return value  # needed for proper sorting
            return str(value)

        return None

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            # special case for index
            if section == 0:
                return self.df.index.name or 'Index'
            return self.df.columns[section - 1]
        return self.df.index[section]

    def flags(self, index):
        if not index.isValid():
            return Qt.ItemIsEnabled

        col = index.column()
        if col == 0:
            return Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable  # index is read-only

        dtype = self.df.dtypes.iat[col - 1]
        flags = Qt.ItemFlags(Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled)

        if self.editable:
            # make boolean columns checkable
            if pd.api.types.is_bool_dtype(dtype):
                flags |= Qt.ItemFlag.ItemIsUserCheckable
            else:
                flags |= Qt.ItemFlag.ItemIsEditable

        return flags

    def setData(self, index, value, role=Qt.EditRole):
        if not index.isValid():
            return False

        col = index.column()
        if col == 0:
            return False  # index is read-only

        row = index.row()
        dtype = self.df.dtypes.iat[col - 1]

        # checkboxes
        if role == Qt.ItemDataRole.CheckStateRole and pd.api.types.is_bool_dtype(dtype):
            self.df.iat[row, col - 1] = value == Qt.Checked
            self.dataChanged.emit(index, index, [Qt.CheckStateRole])
            return True

        if role == Qt.ItemDataRole.EditRole:
            try:
                if not pd.api.types.is_categorical_dtype(dtype):
                    value = dtype.type(value)
                self.df.iat[row, col - 1] = value
            except ValueError:
                return False
            self.dataChanged.emit(index, index, [Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.EditRole])
            return True

        return False

    # custom methods
    @contextmanager
    def changing(self):
        self.layoutAboutToBeChanged.emit()
        yield
        self.layoutChanged.emit()

    def replace_data(self, df):
        with self.changing():
            self.df = df


class DelegateCategorical(QStyledItemDelegate):
    """Delegate that uses comboboxes as editors for categorical data."""

    def createEditor(self, parent, option, index):
        proxy_model = index.model()
        source_model = proxy_model.sourceModel()
        source_index = proxy_model.mapToSource(index)
        col = source_index.column()

        dtype = source_model.df.dtypes.iat[col - 1]

        if pd.api.types.is_categorical_dtype(dtype):
            editor = QComboBox(parent)
            categories = source_model.df.iloc[:, col - 1].cat.categories
            editor.addItems([str(c) for c in categories])
            # allow arrow keys selection
            editor.setFocusPolicy(Qt.StrongFocus)

            # force editor to open on first click, otherwise we need 2 clicks
            QTimer.singleShot(0, editor.showPopup)
            return editor

        return super().createEditor(parent, option, index)

    def setEditorData(self, editor, index):
        if isinstance(editor, QComboBox):
            value = index.model().data(index, Qt.EditRole)
            i = editor.findText(value)
            if i >= 0:
                editor.setCurrentIndex(i)
        else:
            super().setEditorData(editor, index)

    def setModelData(self, editor, model, index):
        if isinstance(editor, QComboBox):
            source_index = model.mapToSource(index)
            source_model = model.sourceModel()
            source_model.setData(
                source_index, editor.currentText(), Qt.EditRole
            )
        else:
            super().setModelData(editor, model, index)


class BoolFriendlyProxyModel(QSortFilterProxyModel):
    """Sort proxy model that handles booleans correctly."""

    def lessThan(self, left, right):
        left_data = self.sourceModel().data(left, Qt.EditRole)
        right_data = self.sourceModel().data(right, Qt.EditRole)

        # ensure booleans compare as expected
        if isinstance(left_data, bool) and isinstance(right_data, bool):
            return left_data < right_data

        return super().lessThan(left, right)


class PandasView(QTableView):
    def __init__(self, parent=None):
        super().__init__(parent)

        # view/model setup with a proxy for sorting and filtering
        proxy_model = BoolFriendlyProxyModel()
        proxy_model.setSourceModel(PandasModel())
        proxy_model.setSortCaseSensitivity(Qt.CaseInsensitive)
        self.setModel(proxy_model)
        self.setSortingEnabled(True)
        # do not auto sort using index on startup
        self.horizontalHeader().setSortIndicator(-1, Qt.AscendingOrder)
        # disable vertical header (since we duplicate it as a column)
        self.verticalHeader().setVisible(False)
        # delegate which provides comboboxes for editing categorical values
        self.setItemDelegate(DelegateCategorical())
        # enable selection and editing
        self.setSelectionBehavior(QTableView.SelectionBehavior.SelectItems)
        self.setSelectionMode(QTableView.ExtendedSelection)
        self.setEditTriggers(QAbstractItemView.AllEditTriggers)

    def keyPressEvent(self, event):
        if event.matches(QKeySequence.Copy):
            self.copySelection()
        elif event.matches(QKeySequence.Paste):
            self.pasteSelection()
        else:
            super().keyPressEvent(event)

    def copySelection(self):
        selection = self.selectedIndexes()
        if not selection:
            return

        proxy_model = self.model()
        model = proxy_model.sourceModel()
        map_func = proxy_model.mapToSource

        # determine selected block
        rows = sorted({idx.row() for idx in selection})
        cols = sorted(
            {idx.column() for idx in selection if idx.column() > 0}
        )  # skip index

        if not rows or not cols:
            return

        # convert proxy indices to actual row/column labels
        actual_rows = [map_func(proxy_model.index(r, 1)).row() for r in rows]
        actual_cols = [c - 1 for c in cols]  # adjust for index

        sub_df = model.df.iloc[actual_rows, actual_cols]

        tsv = sub_df.to_csv(sep='\t', index=False, header=False)
        QGuiApplication.clipboard().setText(tsv)

    def pasteSelection(self):
        proxy_model = self.model()
        model = proxy_model.sourceModel()
        map_func = proxy_model.mapToSource
        df = model.df

        clipboard = QGuiApplication.clipboard().text()
        sel = self.selectedIndexes()

        if not clipboard or not sel or not model.editable:
            return

        rows = clipboard.strip().split('\n')
        data = [row.split('\t') for row in rows]
        n_rows = len(data)
        n_cols = max(len(row) for row in data)

        # copy from top-left selected cell
        start_proxy_index = min(sel, key=lambda x: (x.row(), x.column()))
        start_source_index = map_func(start_proxy_index)
        start_row = start_source_index.row()
        start_col = start_source_index.column() - 1  # skip index column

        if start_col < 0:
            return  # trying to paste into index column — skip

        for i in range(n_rows):
            for j in range(n_cols):
                r = start_row + i
                c = start_col + j
                if r >= df.shape[0] or c >= df.shape[1]:
                    continue
                val = data[i][j]
                dtype = df.dtypes.iloc[c]
                try:
                    if not pd.api.types.is_categorical_dtype(dtype):
                        val = dtype.type(val)
                    df.iat[r, c] = val
                except ValueError:
                    # TODO: warning? undo?
                    pass

        # notify the model/view
        top_left = model.index(start_row, start_col + 1)
        bottom_right = model.index(start_row + n_rows - 1, start_col + n_cols)
        model.dataChanged.emit(top_left, bottom_right)


class FeaturesTable(QWidget):
    """Widget to display layer features as an editable table."""

    def __init__(
        self,
        viewer: napari.viewer.Viewer,
    ) -> None:
        super().__init__()
        self._active_layer = None
        self._selection_blocked = False

        self.viewer = viewer
        self.viewer.layers.selection.events.active.connect(
            self._on_active_layer_change
        )

        self.setLayout(QVBoxLayout())

        self.info = QLabel('')
        self.toggle = QToggleSwitch('editable.')
        self.save = QPushButton('Save as CSV...')
        self.table = PandasView()
        self.layout().addWidget(self.info)
        self.layout().addWidget(self.toggle)
        self.layout().addWidget(self.save)
        self.layout().addWidget(self.table)
        self.layout().addStretch()

        self.toggle.toggled.connect(self._on_editable_change)
        self.save.clicked.connect(self._on_save_clicked)

        self.table.selectionModel().selectionChanged.connect(
            self._on_table_selection_changed
        )

        self._on_active_layer_change()
        self._on_editable_change()

    def _on_active_layer_change(self):
        old_layer = self._active_layer
        self._active_layer = self.viewer.layers.selection.active

        if old_layer is not None and hasattr(old_layer, 'features'):
            old_layer.events.features.disconnect(self._on_features_change)

        if hasattr(self._active_layer, 'features'):
            self._active_layer.events.features.connect(
                self._on_features_change
            )
            if hasattr(self._active_layer, 'selected_label'):
                selection_event = self._active_layer.events.selected_label
            elif hasattr(self._active_layer, 'selected_data'):
                selection_event = self._active_layer.selected_data.events
            selection_event.connect(self._on_layer_selection_changed)

            self._on_layer_selection_changed()
            self._on_features_change()
            self.toggle.setVisible(True)
            self.save.setVisible(True)
            self.table.setVisible(True)
        else:
            self.toggle.setVisible(False)
            self.save.setVisible(False)
            self.table.setVisible(False)

        if self._active_layer is None:
            self.info.setText('No layer selected.')
        elif not hasattr(self._active_layer, 'features'):
            self.info.setText(
                f'"{self._active_layer.name}" has no features table.'
            )
        else:
            self.info.setText(f'Features of "{self._active_layer.name}"')

    def _on_features_change(self):
        # TODO: optimize for smaller changes?
        self.table.model().sourceModel().replace_data(
            self._active_layer.features
        )
        self.table.resizeColumnsToContents()

    def _on_editable_change(self):
        self.table.model().sourceModel().editable = self.toggle.isChecked()

    def _on_table_selection_changed(self):
        """Update layer selection when table cells are selected."""
        if self._selection_blocked:
            return

        if hasattr(self._active_layer, 'selected_label'):
            raw_index = self.table.selectionModel().currentIndex()
            current = self.table.model().mapToSource(raw_index).row()
            self._active_layer.selected_label = current

        elif hasattr(self._active_layer, 'selected_data'):
            selected_rows = {
                self.table.model().mapToSource(raw_index).row()
                for raw_index in self.table.selectionModel().selectedIndexes()
            }
            self._active_layer.selected_data = selected_rows

    def _on_layer_selection_changed(self):
        """Update table selected cells when layer selection changes."""
        model = self.table.model()
        if hasattr(self._active_layer, 'selected_label'):
            sel = self._active_layer.selected_label
            indices = [sel]
        elif hasattr(self._active_layer, 'selected_data'):
            sel = self._active_layer.selected_data
            indices = sel

        selection = QItemSelection()

        for idx in indices:
            idx = model.mapFromSource(model.sourceModel().index(idx, 0))
            # select whole rows
            first_cell = model.index(idx.row(), 0)
            last_cell = model.index(idx.row(), model.columnCount() - 1)
            selection.select(first_cell, last_cell)

        # block signals while setting to prevent infinite loop
        with self._block_selection():
            self.table.selectionModel().select(
                selection,
                QItemSelectionModel.ClearAndSelect | QItemSelectionModel.Rows,
            )

        self.table.viewport().update()

    @contextmanager
    def _block_selection(self):
        self._selection_blocked = True
        yield
        self._selection_blocked = False

    def _on_save_clicked(self):
        dlg = QFileDialog()
        hist = get_save_history()
        dlg.setHistory(hist)

        fname = f'{self._active_layer.name}_features.csv'
        fname = self._remove_invalid_chars(fname)

        fname, _ = dlg.getSaveFileName(
            self,  # parent
            'Save layer features',  # caption
            str(Path(hist[0]) / fname),  # directory in PyQt, dir in PySide
            filter='*.csv',
            options=(
                QFileDialog.DontUseNativeDialog
                if in_ipython()
                else QFileDialog.Options()
            ),
        )

        if not fname:
            # was closed without saving
            return

        df = self.table.model().sourceModel().df
        df.to_csv(fname)

    # copied from QtViewer
    def _remove_invalid_chars(self, selected_layer_name):
        """Removes invalid characters from selected layer name to suggest a filename.

        Parameters
        ----------
        selected_layer_name : str
            The selected napari layer name.

        Returns
        -------
        suggested_name : str
            Suggested name from input selected layer name, without invalid characters.
        """
        unprintable_ascii_chars = (
            '\x00',
            '\x01',
            '\x02',
            '\x03',
            '\x04',
            '\x05',
            '\x06',
            '\x07',
            '\x08',
            '\x0e',
            '\x0f',
            '\x10',
            '\x11',
            '\x12',
            '\x13',
            '\x14',
            '\x15',
            '\x16',
            '\x17',
            '\x18',
            '\x19',
            '\x1a',
            '\x1b',
            '\x1c',
            '\x1d',
            '\x1e',
            '\x1f',
            '\x7f',
        )
        invalid_characters = (
            ''.join(unprintable_ascii_chars)
            + '/'
            + '\\'  # invalid Windows filename character
            + ':*?"<>|\t\n\r\x0b\x0c'  # invalid Windows path characters
        )
        translation_table = dict.fromkeys(map(ord, invalid_characters), None)
        # Remove invalid characters
        suggested_name = selected_layer_name.translate(translation_table)
        return suggested_name
