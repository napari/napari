from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING

import pandas as pd
from qtpy.QtCore import (
    QAbstractTableModel,
    QModelIndex,
    QSortFilterProxyModel,
    Qt,
    QTimer,
)
from qtpy.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QStyledItemDelegate,
    QTableView,
)

if TYPE_CHECKING:
    import napari


class PandasModel(QAbstractTableModel):
    def __init__(self, df=None, parent=None):
        super().__init__(parent)
        self._df = df if df is not None else pd.DataFrame()

    # model methods necessary for qt
    def rowCount(self, parent=None):
        if parent is None:
            parent = QModelIndex()
        return self._df.shape[0]

    def columnCount(self, parent=None):
        if parent is None:
            parent = QModelIndex()
        return self._df.shape[1] + 1  # include index

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None

        row = index.row()
        col = index.column()

        if col == 0:  # index
            if role in (Qt.DisplayRole, Qt.EditRole):
                return str(self._df.index[row])
            return None

        value = self._df.iat[row, col - 1]
        dtype = self._df.dtypes.iat[col - 1]

        # show booleans as respective checkboxes
        if role == Qt.CheckStateRole and pd.api.types.is_bool_dtype(dtype):
            return Qt.Checked if value else Qt.Unchecked

        if role in (Qt.DisplayRole, Qt.EditRole):
            # format based on dtype
            if pd.api.types.is_float_dtype(dtype):
                return f'{value:.6g}'
            if pd.api.types.is_integer_dtype(dtype):
                return f'{value:d}'
            if pd.api.types.is_datetime64_any_dtype(dtype):
                return value.strftime('%Y-%m-%d')
            if pd.api.types.is_bool_dtype(dtype):
                if role == Qt.DisplayRole:
                    return ''  # do not show True/False text
                if role == Qt.EditRole:
                    return value  # needed for proper sorting
            return str(value)

        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal:
            # special case for index
            if section == 0:
                return self._df.index.name or 'Index'
            return self._df.columns[section - 1]
        return self._df.index[section]

    def flags(self, index):
        if not index.isValid():
            return Qt.ItemIsEnabled

        col = index.column()
        if col == 0:
            return Qt.ItemIsEnabled | Qt.ItemIsSelectable  # index is read-only

        dtype = self._df.dtypes.iat[col - 1]
        base_flags = Qt.ItemFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)

        # make boolean columns checkable
        if pd.api.types.is_bool_dtype(dtype):
            return base_flags | Qt.ItemIsUserCheckable

        return base_flags | Qt.ItemIsEditable

    def setData(self, index, value, role=Qt.EditRole):
        if not index.isValid():
            return False

        col = index.column()
        if col == 0:
            return False  # index is read-only

        row = index.row()
        dtype = self._df.dtypes.iat[col - 1]

        # checkboxes
        if role == Qt.CheckStateRole and pd.api.types.is_bool_dtype(dtype):
            self._df.iat[row, col - 1] = value == Qt.Checked
            self.dataChanged.emit(index, index, [Qt.CheckStateRole])
            return True

        if role == Qt.EditRole:
            try:
                if not pd.api.types.is_categorical_dtype(dtype):
                    value = dtype.type(value)
                self._df.iat[row, col - 1] = value
            except ValueError:
                return False
            self.dataChanged.emit(index, index, [Qt.DisplayRole, Qt.EditRole])
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
            self._df = df


class DelegateCategorical(QStyledItemDelegate):
    def createEditor(self, parent, option, index):
        proxy_model = index.model()
        source_model = proxy_model.sourceModel()
        source_index = proxy_model.mapToSource(index)
        col = source_index.column()

        dtype = source_model._df.dtypes.iat[col - 1]

        if pd.api.types.is_categorical_dtype(dtype):
            editor = QComboBox(parent)
            categories = source_model._df.iloc[:, col - 1].cat.categories
            editor.addItems([str(c) for c in categories])
            # allow arrow keys selection
            editor.setFocusPolicy(Qt.StrongFocus)

            # needed to delay opening until editor is fully created
            QTimer.singleShot(0, editor.showPopup)
            return editor

        return super().createEditor(parent, option, index)

    def editorEvent(self, event, model, option, index):
        # force editor to open on first click, otherwise we need 2 clicks
        if event.type() == event.MouseButtonPress:
            view = option.widget
            view.edit(index)
        return super().editorEvent(event, model, option, index)

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
    def lessThan(self, left, right):
        left_data = self.sourceModel().data(left, Qt.EditRole)
        right_data = self.sourceModel().data(right, Qt.EditRole)

        # ensure booleans compare as expected
        if isinstance(left_data, bool) and isinstance(right_data, bool):
            return not left_data and right_data

        return super().lessThan(left, right)


class FeaturesTable(QTableView):
    def __init__(
        self,
        viewer: napari.viewer.Viewer,
    ) -> None:
        super().__init__()
        self._active_layer = None

        self.viewer = viewer
        self.viewer.layers.selection.events.active.connect(
            self._on_active_layer_change
        )

        proxy_model = BoolFriendlyProxyModel()
        proxy_model.setSourceModel(PandasModel())
        proxy_model.setSortCaseSensitivity(Qt.CaseInsensitive)
        self.setModel(proxy_model)
        self.setSortingEnabled(True)
        # do not auto sort using index on startup
        self.horizontalHeader().setSortIndicator(-1, Qt.AscendingOrder)
        self.setEditTriggers(QAbstractItemView.AllEditTriggers)
        self.setItemDelegate(DelegateCategorical())
        self.verticalHeader().setVisible(False)
        self._on_active_layer_change()

    def _on_active_layer_change(self):
        if self._active_layer is not None:
            self._active_layer.events.features.disconnect(
                self._on_features_change
            )

        self._active_layer = self.viewer.layers.selection.active

        if hasattr(self._active_layer, 'features'):
            self._active_layer.events.features.connect(
                self._on_features_change
            )
            self._on_features_change()

    def _on_features_change(self):
        # TODO: optimize for smaller changes?
        self.model().sourceModel().replace_data(self._active_layer.features)
        self.resizeColumnsToContents()
