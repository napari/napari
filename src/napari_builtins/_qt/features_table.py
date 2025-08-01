from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
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
    import napari.components


class PandasModel(QAbstractTableModel):
    """Qt Model for a pandas DataFrame.

    Follows the Qt Model/View protocol, implementing all the needed methods
    for a QTableView to be able to read/write changes to a pandas dataframe.
    Columns of specific dtypes get special treatment (bools become tickboxes,
    categoricals are exposed through comboboxes, etc).
    It's designed to be used in conjunction with the BoolFriendlyProxyModel
    in order to properly sort, and with the DelegateCategorical in order to
    provide comboboxes for categoricals.
    """

    def __init__(self, df: pd.DataFrame | None = None, parent=None):
        super().__init__(parent)
        self.df = df if df is not None else pd.DataFrame()
        self.editable = False

    # model methods necessary for qt
    def rowCount(self, parent=None):
        return self.df.shape[0]

    def columnCount(self, parent=None):
        return self.df.shape[1] + 1  # include index

    def data(self, index: QModelIndex, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():  # pragma: no cover
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
        if (
            role == Qt.ItemDataRole.CheckStateRole
            and pd.api.types.is_bool_dtype(dtype)
        ):
            return Qt.CheckState.Checked if value else Qt.CheckState.Unchecked

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

    def headerData(
        self,
        section: int,
        orientation: Qt.Orientation,
        role=Qt.ItemDataRole.DisplayRole,
    ) -> Any:
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            # special case for index
            if section == 0:
                return self.df.index.name or 'Index'
            return self.df.columns[section - 1]
        return self.df.index[section]

    def flags(self, index: QModelIndex) -> Qt.ItemFlag:
        if not index.isValid():
            return Qt.ItemFlag.ItemIsEnabled

        col = index.column()
        # index and layer_name columns are read-only

        flags = Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled
        if col == 0 or col == self.df.shape[1]:  # index or layer_name column
            return flags

        dtype = self.df.dtypes.iat[col - 1]

        if self.editable:
            # make boolean columns checkable
            if pd.api.types.is_bool_dtype(dtype):
                flags |= Qt.ItemFlag.ItemIsUserCheckable
            else:
                flags |= Qt.ItemFlag.ItemIsEditable

        return flags

    def setData(
        self, index: QModelIndex, value: Any, role=Qt.ItemDataRole.EditRole
    ) -> bool:
        if not index.isValid():
            return False

        col = index.column()
        if col == 0:
            return False  # index is read-only
        if col == self.df.shape[1]:
            return False  # layer_name column is read-only

        row = index.row()
        dtype = self.df.dtypes.iat[col - 1]

        # checkboxes
        if (
            role == Qt.ItemDataRole.CheckStateRole
            and pd.api.types.is_bool_dtype(dtype)
        ):
            self.df.iat[row, col - 1] = (
                Qt.CheckState(value) == Qt.CheckState.Checked
            )
            self.dataChanged.emit(
                index, index, [Qt.ItemDataRole.CheckStateRole]
            )
            return True

        if role == Qt.ItemDataRole.EditRole:
            try:
                if not isinstance(dtype, pd.CategoricalDtype):
                    value = dtype.type(value)
                self.df.iat[row, col - 1] = value
            except (TypeError, ValueError):
                # Type error is for categorical types that cannot be converted
                # Value error is for invalid values in dtype.type(value)
                return False
            self.dataChanged.emit(
                index,
                index,
                [Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.EditRole],
            )
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

    def createEditor(self, parent, option, index: QModelIndex) -> QWidget:
        proxy_model = index.model()
        source_model = proxy_model.sourceModel()
        source_index = proxy_model.mapToSource(index)
        col = source_index.column()

        dtype = source_model.df.dtypes.iat[col - 1]

        if isinstance(dtype, pd.CategoricalDtype):
            editor = QComboBox(parent)
            categories = source_model.df.iloc[:, col - 1].cat.categories
            editor.addItems([str(c) for c in categories])
            # allow arrow keys selection
            editor.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

            # force editor to open on first click, otherwise we need 2 clicks
            QTimer.singleShot(0, editor.showPopup)
            return editor

        return super().createEditor(parent, option, index)

    def setEditorData(self, editor: QWidget, index: QModelIndex):
        if isinstance(editor, QComboBox):
            value = index.model().data(index, Qt.ItemDataRole.EditRole)
            i = editor.findText(value)
            if i >= 0:
                editor.setCurrentIndex(i)
        else:
            super().setEditorData(editor, index)

    def setModelData(
        self, editor: QWidget, model: QSortFilterProxyModel, index: QModelIndex
    ):
        if isinstance(editor, QComboBox):
            source_index = model.mapToSource(index)
            source_model = model.sourceModel()
            source_model.setData(
                source_index, editor.currentText(), Qt.ItemDataRole.EditRole
            )
        else:
            super().setModelData(editor, model, index)


class BoolFriendlyProxyModel(QSortFilterProxyModel):
    """Sort proxy model that handles booleans correctly."""

    def lessThan(self, left: Any, right: Any) -> bool:
        left_data = self.sourceModel().data(left, Qt.ItemDataRole.EditRole)
        right_data = self.sourceModel().data(right, Qt.ItemDataRole.EditRole)

        # ensure booleans compare as expected. Not sure what happens internally in qt
        # that doesn't work, but doing it ourselves in python works.
        # One thing that breaks it are numpy bools
        if all(pd.api.types.is_bool(d) for d in (left_data, right_data)):
            return bool(left_data < right_data)

        return super().lessThan(left, right)


class PandasView(QTableView):
    """View for PandasModel to interact with pandas dataframes via a table.

    This class is designed to work in tandem with PandasModel, and allows
    the following things:
    - sorting the table without affecting the data
    - copy/pasting cells
    - edting the data with specific editors depending on the dtype
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        # view/model setup with a proxy for sorting and filtering
        proxy_model = BoolFriendlyProxyModel()
        proxy_model.setSourceModel(PandasModel())
        proxy_model.setSortCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self.setModel(proxy_model)
        self.setSortingEnabled(True)
        # do not auto sort using index on startup
        self.horizontalHeader().setSortIndicator(
            -1, Qt.SortOrder.AscendingOrder
        )
        # disable vertical header (since we duplicate it as a column)
        self.verticalHeader().setVisible(False)
        # delegate which provides comboboxes for editing categorical values
        self.setItemDelegate(DelegateCategorical())
        # enable selection and editing
        self.setSelectionBehavior(QTableView.SelectionBehavior.SelectItems)
        self.setSelectionMode(QTableView.SelectionMode.ExtendedSelection)
        self.setEditTriggers(QAbstractItemView.EditTrigger.AllEditTriggers)

    def keyPressEvent(self, event):
        if event.matches(QKeySequence.StandardKey.Copy):
            self.copySelection()
        elif event.matches(QKeySequence.StandardKey.Paste):
            self.pasteSelection()
        else:
            super().keyPressEvent(event)

    def copySelection(self):
        selection = self.selectedIndexes()
        if not selection:
            return  # pragma: no cover

        proxy_model = self.model()
        model = proxy_model.sourceModel()
        map_func = proxy_model.mapToSource

        # determine selected block (excluding index and layer_name columns)
        rows = sorted(
            {
                idx.row()
                for idx in selection
                if 0 < idx.column() < model.df.shape[1]
            }
        )
        cols = sorted(
            {
                idx.column()
                for idx in selection
                if 0 < idx.column() < model.df.shape[1]
            }
        )

        if not rows or not cols:
            return  # pragma: no cover

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

        # if index or layer_name column is in the selection, just get out
        if (
            not clipboard
            or not sel
            or not model.editable
            or any(
                s.column() == 0 or s.column() == model.df.shape[1] for s in sel
            )
        ):
            return  # pragma: no cover

        rows = clipboard.strip().split('\n')
        data = [row.split('\t') for row in rows]
        n_rows = len(data)
        n_cols = max(len(row) for row in data)

        # copy from top-left selected cell
        start_proxy_index = min(sel, key=lambda x: (x.row(), x.column()))
        start_source_index = map_func(start_proxy_index)
        start_row = start_source_index.row()
        start_col = start_source_index.column()

        for i in range(n_rows):
            for j in range(n_cols):
                r = start_row + i
                view_col = start_col + j  # view column index
                c = view_col - 1  # dataframe column index (offset for index)
                if (
                    r >= df.shape[0]
                    or view_col >= model.columnCount()
                    or view_col == 0
                    or view_col == model.df.shape[1]
                ):  # skip index or layer_name columns
                    continue
                val = data[i][j]
                dtype = df.dtypes.iloc[c]
                try:
                    if not isinstance(dtype, pd.CategoricalDtype):
                        val = dtype.type(val)
                    df.iat[r, c] = val
                except (ValueError, TypeError):  # pragma: no cover
                    # TODO: warning? undo?
                    pass

        # notify the model/view
        top_left = model.index(start_row, start_col)
        bottom_right = model.index(start_row + n_rows - 1, start_col + n_cols)
        model.dataChanged.emit(top_left, bottom_right)


class FeaturesTable(QWidget):
    """Widget to display layer features as an editable table.

    The table automatically shows the contents of the current active layer
    in the layerlist, if that layer has a features table.
    Features can be edited directly if the `editable` toggle is enabled.
    Columns can be sorted without affecting the layer data.
    Selecting a row in the table will select the corresponding data in the
    layer, and viceversa.
    Data can be copy/pasted as csv, and can be saved to file.
    """

    def __init__(
        self,
        viewer: napari.viewer.ViewerModel,
    ) -> None:
        super().__init__()
        self._selected_layers = []
        self._selection_blocked = False

        self.viewer = viewer
        self.viewer.layers.selection.events.changed.connect(
            self._on_layer_selection_change
        )

        self.setLayout(QVBoxLayout())

        self.info = QLabel('')
        self.toggle = QToggleSwitch('editable.')
        self.join_toggle = QToggleSwitch('common_columns.')
        self.save = QPushButton('Save as CSV...')
        self.table = PandasView()
        self.layout().addWidget(self.info)
        self.layout().addWidget(self.toggle)
        self.layout().addWidget(self.join_toggle)
        self.layout().addWidget(self.save)
        self.layout().addWidget(self.table)
        self.layout().addStretch()

        self.toggle.toggled.connect(self._on_editable_change)
        self.join_toggle.toggled.connect(self._on_join_change)
        self.save.clicked.connect(self._on_save_clicked)

        self.table.selectionModel().selectionChanged.connect(
            self._on_table_selection_changed
        )

        self._on_layer_selection_change()
        self._on_editable_change()

    @staticmethod
    def _get_selection_event_for_layer(layer):
        if hasattr(layer, 'selected_label'):
            return layer.events.selected_label
        if hasattr(layer, 'selected_data'):
            # Points layer has selected_data.events, but Shapes layer uses highlight event
            if hasattr(layer.selected_data, 'events'):
                return layer.selected_data.events
            if hasattr(layer.events, 'highlight'):
                return layer.events.highlight
        # Return None if layer doesn't have expected selection attributes
        return None

    def _on_layer_selection_change(self):
        """Update the table when the layer selection changes and handles event connections."""
        old_layer_list = self._selected_layers
        # Filter to only keep layers with features
        self._selected_layers = [
            layer
            for layer in self.viewer.layers.selection
            if hasattr(layer, 'features')
        ]

        if len(old_layer_list) > 0:
            # disconnect events from old layers
            for layer in old_layer_list:
                layer.events.features.disconnect(self._on_features_change)
                selection_event = self._get_selection_event_for_layer(layer)
                if selection_event is not None:
                    selection_event.disconnect(
                        self._update_table_selected_cells
                    )

        if len(self._selected_layers) > 0:
            # connect events to new layers (all have features by definition)
            for layer in self._selected_layers:
                layer.events.features.connect(self._on_features_change)
                selection_event = self._get_selection_event_for_layer(layer)
                if selection_event is not None:
                    selection_event.connect(self._update_table_selected_cells)

            # Show widgets and update table
            self._on_features_change()
            self.toggle.setVisible(True)
            self.save.setVisible(True)
            self.table.setVisible(True)
            self.join_toggle.setVisible(len(self._selected_layers) > 1)
            self.info.setText(
                f'Features of "{[layer.name for layer in self._selected_layers]}"'
            )
        else:
            # Hide widgets and show appropriate message
            self.toggle.setVisible(False)
            self.join_toggle.setVisible(False)
            self.save.setVisible(False)
            self.table.setVisible(False)

            # Determine message based on original selection
            if len(self.viewer.layers.selection) > 0:
                self.info.setText('Selected layers do not have features.')
            else:
                self.info.setText('No layer selected.')

    def _on_features_change(self):
        """Update the table with the features of the currently selected layers."""
        # TODO: optimize for smaller changes?
        join_type = 'inner' if self.join_toggle.isChecked() else 'outer'
        self.table.model().sourceModel().replace_data(
            self._build_multilayer_features_table(join=join_type)
        )
        self.table.resizeColumnsToContents()
        self._update_table_selected_cells()

    def _build_multilayer_features_table(
        self, join: str = 'outer'
    ) -> pd.DataFrame:
        """Builds a features table for multiple layers."""
        df_list = []
        for layer in self._selected_layers:
            # All layers in self._selected_layers are guaranteed to have features
            if layer.features is not None:
                if 'layer' not in layer.features.columns:
                    layer.features['layer'] = layer.name
                    layer.features['layer'] = layer.features['layer'].astype(
                        'category'
                    )
                df_list.append(layer.features)
        df = pd.concat(df_list, ignore_index=True, join=join)
        # ensure 'layer' is the last column
        layer_name_col = df.pop('layer')
        df['layer'] = layer_name_col
        return df

    def _on_editable_change(self):
        self.table.model().sourceModel().editable = self.toggle.isChecked()

    def _on_join_change(self):
        """Update the table when join mode changes."""
        self._on_features_change()

    def _on_table_selection_changed(self):
        """Update layer selection when table cells are selected."""
        if self._selection_blocked:
            return

        # Get all selected row indices from the table
        selected_global_rows = [
            self.table.model().mapToSource(raw_index).row()
            for raw_index in self.table.selectionModel().selectedIndexes()
        ]

        if not selected_global_rows:
            return

        df = self.table.model().sourceModel().df
        if df.empty:
            return

        # Calculate layer start indices for all layers once (most efficient)
        layer_starts = df.groupby('layer', sort=False, observed=False).apply(
            lambda x: x.index[0], include_groups=False
        )

        # Get layer names for selected rows and convert to layer-specific indices
        selected_layer_names = df['layer'].iloc[selected_global_rows]
        layer_start_indices = selected_layer_names.map(layer_starts).astype(
            int
        )
        layer_specific_indices = (
            np.array(selected_global_rows) - layer_start_indices.values
        )

        # Group by layer name efficiently
        selections_by_layer = {}
        for layer_name, layer_idx in zip(
            selected_layer_names, layer_specific_indices, strict=False
        ):
            selections_by_layer.setdefault(layer_name, []).append(layer_idx)

        # Update layer selections
        for layer in self._selected_layers:
            if layer.name not in selections_by_layer:
                continue
            layer_indices = selections_by_layer[layer.name]

            with self._block_selection():
                if hasattr(layer, 'selected_label'):
                    layer.selected_label = layer_indices[-1]
                elif hasattr(layer, 'selected_data'):
                    layer.selected_data = set(layer_indices)

    def _update_table_selected_cells(self):
        """Update table selected cells when layer selection changes."""
        if self._selection_blocked:
            return

        model = self.table.model()
        df = model.sourceModel().df
        if df.empty:
            return

        indices = []
        for layer in self._selected_layers:
            if hasattr(layer, 'selected_label'):
                sel = layer.selected_label
                layer_data_row_index = df[df['layer'] == layer.name].index
                indices += [[layer_data_row_index[sel]]]
            elif hasattr(layer, 'selected_data'):
                sel = layer.selected_data
                layer_data_row_index = df[df['layer'] == layer.name].index
                indices += [layer_data_row_index[list(sel)]]
            else:
                continue

        if not indices:
            return  # No layers with selection attributes, nothing to select

        indices = np.concatenate(indices)
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
                QItemSelectionModel.SelectionFlag.ClearAndSelect
                | QItemSelectionModel.SelectionFlag.Rows,
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

        # Generate filename for multiple layers
        if len(self._selected_layers) == 1:
            fname = f'{self._selected_layers[0].name}_features.csv'
        elif len(self._selected_layers) > 1:
            first_layer = self._selected_layers[0].name
            others_count = len(self._selected_layers) - 1
            fname = (
                f'{first_layer}_and_{others_count}_other_layers_features.csv'
            )

        fname = self._remove_invalid_chars(fname)

        fname, _ = dlg.getSaveFileName(
            self,  # parent
            'Save layer features',  # caption
            str(Path(hist[0]) / fname),  # directory in PyQt, dir in PySide
            filter='*.csv',
            options=(
                QFileDialog.Option.DontUseNativeDialog
                if in_ipython()
                else QFileDialog.Options()
            ),
        )

        if not fname:
            # was closed without saving
            return  # pragma: no cover

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
