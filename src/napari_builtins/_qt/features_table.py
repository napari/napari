from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any
from warnings import warn

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

    This model supports the concept of immutable columns identified by name, not position.
    For pandas DataFrames, column 0 (representing the DataFrame index) is automatically
    treated as immutable.

    Parameters
    ----------
    df : pd.DataFrame, optional
        The pandas DataFrame to wrap.
    immutable_columns : list[str], optional
        List of column names to be treated as immutable (read-only).
    parent : QObject, optional
        Parent Qt object.
    """

    def __init__(
        self,
        df: pd.DataFrame | None = None,
        immutable_columns: list[str] | None = None,
        source_column_name: str | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self.df = df if df is not None else pd.DataFrame()
        self.editable = False
        self._immutable_columns = set(immutable_columns or [])
        self._source_column_name = source_column_name
        self._original_categories: dict[
            str, dict[str, list]
        ] = {}  # {layer_name: {col_name: [categories]}}
        self._add_index_to_immutable()

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
        value, dtype = self._get_cell_value_and_dtype(row, col)

        # show booleans as respective checkboxes
        if (
            role == Qt.ItemDataRole.CheckStateRole
            and pd.api.types.is_bool_dtype(dtype)
        ):
            return Qt.CheckState.Checked if value else Qt.CheckState.Unchecked

        if role in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.EditRole):
            if pd.api.types.is_float_dtype(dtype):
                return float(value)
            if pd.api.types.is_integer_dtype(dtype):
                return int(value)
            if isinstance(
                value, pd.Timestamp
            ) or pd.api.types.is_datetime64_any_dtype(dtype):
                return value.strftime('%Y-%m-%d')
            if pd.api.types.is_bool_dtype(dtype):
                if role == Qt.ItemDataRole.DisplayRole:
                    return ''  # do not show True/False text
                if role == Qt.ItemDataRole.EditRole:
                    return bool(value)  # needed for proper sorting
            if value is pd.NA:
                return pd.NA  # shows empty cell in the view
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

        if isinstance(self.df, pd.DataFrame):
            if orientation == Qt.Orientation.Horizontal:
                if self._has_source_column():
                    # With source column: col 0 = Source, col 1 = Index, col 2+ = features
                    if section == 0:
                        return self._source_column_name
                    if section == 1:
                        return self.df.index.name or 'Index'
                    return self.df.columns[section - 1]  # Offset by 1
                # Without source column: col 0 = Index, col 1+ = features
                if section == 0:
                    return self.df.index.name or 'Index'
                return self.df.columns[section - 1]
            # Vertical header
            return self.df.index[section]
        # TODO: For non-pandas dataframes, implement appropriate header handling
        return str(section)

    def flags(self, index: QModelIndex) -> Qt.ItemFlag:
        if not index.isValid():
            return Qt.ItemFlag.ItemIsEnabled

        row = index.row()
        col = index.column()
        value, dtype = self._get_cell_value_and_dtype(row, col)

        # Check if this column is immutable
        flags = Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled
        if self.is_column_immutable(col):
            return flags

        # Cells with pd.NA are not editable
        if value is pd.NA:
            return flags

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
        # Check if this column is immutable
        if self.is_column_immutable(col):
            return False

        row = index.row()
        # Get DataFrame column index
        df_col = self._get_df_col_idx(col)
        if df_col is None:
            # Index or Layer column - should be caught by immutability check
            return False
        dtype = np.dtype(type(self.df.iat[row, df_col]))

        # checkboxes
        if (
            role == Qt.ItemDataRole.CheckStateRole
            and pd.api.types.is_bool_dtype(dtype)
        ):
            self.df.iat[row, df_col] = (
                Qt.CheckState(value) == Qt.CheckState.Checked
            )
            self.dataChanged.emit(
                index, index, [Qt.ItemDataRole.CheckStateRole]
            )
            return True

        if role == Qt.ItemDataRole.EditRole:
            # Warn if setting a string cell to a number-like value
            if pd.api.types.is_string_dtype(dtype) and self._looks_like_number(
                str(value)
            ):
                warn(
                    f'Setting a string cell to the value "{value}" which looks like a number.',
                    UserWarning,
                    stacklevel=3,
                )

            try:
                if not isinstance(dtype, pd.CategoricalDtype):
                    value = dtype.type(value)
                self.df.iat[row, df_col] = value
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
            self._add_index_to_immutable()
            # Reset provenance when data is replaced
            self._original_categories = {}

    def set_original_categories(
        self, categories_dict: dict[str, dict[str, list]]
    ) -> None:
        """Store original categories by layer before concat conversion to object dtype.

        Parameters
        ----------
        categories_dict : dict[str, dict[str, list]]
            Mapping of layer_name -> {column_name -> [category_values]}
            Only includes columns that were categorical in each layer.
        """
        self._original_categories = categories_dict

    def is_categorical_in_layer(self, col_name: str, layer_name: str) -> bool:
        """Check if a column was categorical in a specific layer.

        Parameters
        ----------
        col_name : str
            Column name to check
        layer_name : str
            Layer name to check against

        Returns
        -------
        bool
            True if column was categorical in that layer before concat
        """
        return col_name in self._original_categories.get(layer_name, {})

    def get_original_categories(
        self, col_name: str, layer_name: str
    ) -> list | None:
        """Get original categories for a column in a specific layer.

        Parameters
        ----------
        col_name : str
            Column name
        layer_name : str
            Layer name

        Returns
        -------
        list | None
            List of categories if column was categorical in that layer, else None
        """
        return self._original_categories.get(layer_name, {}).get(col_name)

    def set_immutable_columns(self, column_names: list[str]):
        """Set columns that should be treated as immutable (read-only).

        Parameters
        ----------
        column_names : list[str]
            List of column names to mark as immutable. The index column (0)
            is always immutable regardless of this setting.
        """
        self._immutable_columns = set(column_names)
        self._add_index_to_immutable()

    def get_immutable_columns(self) -> list[str]:
        """Get the list of column names that are marked as immutable.

        Returns
        -------
        list[str]
            List of column names that are immutable (read-only).
        """
        return list(self._immutable_columns)

    def _add_index_to_immutable(self):
        """Add the index name (or 'Index') to the immutable columns if a pandas DataFrame."""
        if isinstance(self.df, pd.DataFrame):
            if self.df.index.name is not None:
                self._immutable_columns.add(self.df.index.name)
            else:
                self._immutable_columns.add('Index')

    def _has_source_column(self) -> bool:
        """Check if the DataFrame has a source identifier column.

        Returns
        -------
        bool
            True if source column exists in the DataFrame.
        """
        return (
            self._source_column_name is not None
            and self._source_column_name in self.df.columns
        )

    def _get_df_col_idx(self, view_col: int) -> int | None:
        """Convert view column index to DataFrame column index.

        When a source column exists, the view layout is:
        - view col 0 → source column (e.g., 'Layer') at df.columns[0]
        - view col 1 → DataFrame index
        - view col 2+ → DataFrame columns starting at df.columns[1]

        Without a source column, the view layout is:
        - view col 0 → DataFrame index
        - view col 1+ → DataFrame columns starting at df.columns[0]

        Parameters
        ----------
        view_col : int
            The column index from the view

        Returns
        -------
        int | None
            DataFrame column index, or None if this is the index column or source column
        """
        if self._has_source_column():
            # Multi-source mode: Source, Index, then features
            if view_col == 0:
                return (
                    None  # Source column (special handling needed by caller)
                )
            if view_col == 1:
                return None  # Index column
            return view_col - 1  # view col 2 → df col 1, etc.
        # Single-source mode: Index, then features
        if view_col == 0:
            return None  # Index column
        return view_col - 1  # view col 1 → df col 0, etc.

    def _get_cell_value_and_dtype(
        self, row: int, col: int
    ) -> tuple[Any, np.dtype]:
        """Get value and dtype for a cell at the given row and column.

        Parameters
        ----------
        row : int
            Row index in the model
        col : int
            Column index in the model (view column)

        Returns
        -------
        tuple[Any, np.dtype]
            The cell value and its dtype
        """
        if self._has_source_column():
            # With source column: col 0 = Source, col 1 = Index, col 2+ = features
            if col == 0:
                value = self.df.iat[row, 0]  # Source column is first in df
            elif col == 1:
                value = self.df.index[row]
            else:
                value = self.df.iat[
                    row, col - 1
                ]  # Offset by 1 (source column)
        else:
            # Without source column: col 0 = Index, col 1+ = features
            if col == 0:
                value = self.df.index[row]
            else:
                value = self.df.iat[row, col - 1]

        return value, np.dtype(type(value))

    def is_column_immutable(self, col_idx: int) -> bool:
        """Check if a column is immutable based on its index.

        For pandas DataFrames:
        - With source column: columns 0 (source) and 1 (Index) are immutable
        - Without source column: column 0 (Index) is immutable
        Columns with names in self._immutable_columns are also immutable.

        Parameters
        ----------
        col_idx : int
            Column index to check

        Returns
        -------
        bool
            True if the column is immutable, False otherwise
        """
        if not isinstance(self.df, pd.DataFrame):
            return False

        # Check if this is an immutable structural column (source or index)
        if self._has_source_column():
            # With source: Source (col 0) and Index (col 1) are immutable
            if col_idx in (0, 1):
                return True
        else:
            # Without source: Index (col 0) is immutable
            if col_idx == 0:
                return True

        # Check if the feature column name is in immutable set
        if col_idx - 1 < len(self.df.columns):
            col_name = self.df.columns[col_idx - 1]
            return col_name in self._immutable_columns

        return False

    @staticmethod
    def _looks_like_number(value: str) -> bool:
        """Check if a string looks like a number (int or float).

        Parameters
        ----------
        value : str
            The string value to check

        Returns
        -------
        bool
            True if the string looks like an int or float, False otherwise
        """
        if not isinstance(value, str):
            return False
        stripped = value.strip()
        if not stripped:
            return False
        try:
            float(stripped)
        except ValueError:
            return False
        else:
            return True


class DelegateCategorical(QStyledItemDelegate):
    """Delegate that uses comboboxes as editors for categorical data."""

    def createEditor(self, parent, option, index: QModelIndex) -> QWidget:
        proxy_model = index.model()
        source_model = proxy_model.sourceModel()
        source_index = proxy_model.mapToSource(index)
        col = source_index.column()

        # Get value via proxy_model for EditRole
        value = proxy_model.data(index, Qt.ItemDataRole.EditRole)
        dtype = np.dtype(type(value))
        # If bool dtype or value is bool, let Qt use default checkbox editor
        if pd.api.types.is_bool_dtype(dtype) or isinstance(value, bool):
            return None

        # Map view column to DataFrame column name
        df_col_idx = source_model._get_df_col_idx(col)
        if df_col_idx is None:
            # Index or Layer column - no editor needed
            return None
        col_name = source_model.df.columns[df_col_idx]

        # Determine layer name for this row
        if source_model._has_source_column():
            # Multi-source case: get source from the source column
            layer_name = source_model.df.iloc[source_index.row()][
                source_model._source_column_name
            ]
        else:
            # Single-layer case: get the layer name from provenance (should be exactly one)
            layer_names = list(source_model._original_categories.keys())
            layer_name = layer_names[0] if layer_names else None

        # Check provenance: only create combobox if column was originally categorical
        original_categories = None
        if layer_name is not None:
            original_categories = source_model.get_original_categories(
                col_name, layer_name
            )

        if original_categories is not None:
            editor = QComboBox(parent)
            editor.addItems([str(c) for c in original_categories])
            # allow arrow keys selection
            editor.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

            # force editor to open on first click, otherwise we need 2 clicks
            QTimer.singleShot(0, editor.showPopup)
            return editor

        # Fallback: check current dtype for backward compatibility
        if isinstance(dtype, pd.CategoricalDtype):
            editor = QComboBox(parent)
            categories = source_model.df.iloc[:, df_col_idx].cat.categories
            editor.addItems([str(c) for c in categories])
            # allow arrow keys selection
            editor.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

            # force editor to open on first click, otherwise we need 2 clicks
            QTimer.singleShot(0, editor.showPopup)
            return editor

        # If float, use spinbox
        if pd.api.types.is_float_dtype(dtype):
            editor = super().createEditor(parent, option, index)
            editor.setDecimals(10)
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
    """Sort proxy model that handles booleans correctly and sorts immutable columns as integers if possible."""

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
        proxy_model.setSourceModel(PandasModel(source_column_name='Layer'))
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

        # determine selected block (excluding immutable columns)
        model = self.model().sourceModel()

        # Filter out immutable columns
        rows = sorted(
            {
                idx.row()
                for idx in selection
                if not model.is_column_immutable(idx.column())
            }
        )
        cols = sorted(
            {
                idx.column()
                for idx in selection
                if not model.is_column_immutable(idx.column())
            }
        )

        if not rows or not cols:
            return  # pragma: no cover

        # convert proxy indices to actual row/column labels
        actual_rows = [map_func(proxy_model.index(r, 1)).row() for r in rows]
        # Map view columns to df columns, filtering out immutable columns (None)
        actual_cols = [model._get_df_col_idx(c) for c in cols]
        actual_cols = [c for c in actual_cols if c is not None]

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
        # if any immutable column is in the selection, just get out
        if (
            not clipboard
            or not sel
            or not model.editable
            or any(model.is_column_immutable(s.column()) for s in sel)
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

        # Track max modified column
        max_modified_col = start_col

        for i in range(n_rows):
            for j in range(n_cols):
                r = start_row + i
                c = start_col + j
                # Get DataFrame column index from source model column
                df_col = model._get_df_col_idx(c)
                if df_col is None:  # Index or Layer column - skip
                    continue
                if r >= df.shape[0] or df_col >= df.shape[1]:
                    continue
                val = data[i][j]
                dtype = df.dtypes.iloc[df_col]
                try:
                    if not isinstance(dtype, pd.CategoricalDtype):
                        val = dtype.type(val)
                    df.iat[r, df_col] = val
                    # Track the max column that was actually modified
                    max_modified_col = max(max_modified_col, c)
                except (ValueError, TypeError):  # pragma: no cover
                    # TODO: warning? undo?
                    pass

        # notify the model/view
        top_left = model.index(start_row, start_col)
        bottom_right = model.index(start_row + n_rows - 1, max_modified_col)
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
        self.viewer.layers.events.renamed.connect(self._on_layer_renamed)

        self.setLayout(QVBoxLayout())

        self.info = QLabel('')
        self.toggle = QToggleSwitch('editable')
        self.join_toggle = QToggleSwitch('shared columns only')
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

        # Connect to dataChanged to sync edits back to layers
        self.table.model().sourceModel().dataChanged.connect(
            self._on_table_data_changed
        )

        self._on_layer_selection_change()
        self._on_editable_change()

    def _on_layer_renamed(self, event):
        """Update table and info label when a layer is renamed."""
        if not self._selected_layers:
            return

        # Update info label with new layer names
        if len(self._selected_layers) == 1:
            self.info.setText(f'Features of "{self._selected_layers[0].name}"')
        else:
            layer_names = ', '.join(
                f'"{layer.name}"'
                for layer in sorted(
                    self._selected_layers, key=lambda lyr: lyr.name
                )
            )
            self.info.setText(f'Features of [{layer_names}]')

        # Update the Layer column in the table if present
        model = self.table.model().sourceModel()
        df = model.df
        if 'Layer' in df.columns and len(self._selected_layers) > 1:
            # Rebuild the table to reflect the renamed layer
            self._on_features_change()

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

    def _disconnect_layer_events(self, layers):
        """Disconnect features and selection events from the update table callbacks."""
        for layer in layers:
            if hasattr(layer, 'events') and hasattr(layer.events, 'features'):
                layer.events.features.disconnect(self._on_features_change)
            selection_event = self._get_selection_event_for_layer(layer)
            if selection_event is not None:
                selection_event.disconnect(self._update_table_selected_cells)

    def _connect_layer_events(self, layers):
        """Connect features and selection events to the appropriate update table callbacks."""
        for layer in layers:
            if hasattr(layer, 'events') and hasattr(layer.events, 'features'):
                layer.events.features.connect(self._on_features_change)
            selection_event = self._get_selection_event_for_layer(layer)
            if selection_event is not None:
                selection_event.connect(self._update_table_selected_cells)

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
            self._disconnect_layer_events(old_layer_list)

        if len(self._selected_layers) > 0:
            self._connect_layer_events(self._selected_layers)

            # Show widgets and update table
            self._on_features_change()
            self.toggle.setVisible(True)
            self.save.setVisible(True)
            self.table.setVisible(True)
            self.join_toggle.setVisible(len(self._selected_layers) > 1)
            if len(self._selected_layers) == 1:
                self.info.setText(
                    f'Features of "{self._selected_layers[0].name}"'
                )
            else:
                layer_names = ', '.join(
                    f'"{layer.name}"'
                    for layer in sorted(
                        self._selected_layers, key=lambda lyr: lyr.name
                    )
                )
                self.info.setText(f'Features of [{layer_names}]')
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

        # Extract categories before concat converts them to object dtype
        original_categories = self._get_original_categories_from_layers()

        df = self._build_multilayer_features_table(join=join_type)

        # Replace data and configure immutable columns
        model = self.table.model().sourceModel()
        model.replace_data(df)
        model.set_original_categories(original_categories)
        model.set_immutable_columns(['Layer'] if 'Layer' in df.columns else [])

        self.table.resizeColumnsToContents()
        self._update_table_selected_cells()

    def _build_multilayer_features_table(
        self, join: str = 'outer'
    ) -> pd.DataFrame:
        """Builds a features table for multiple layers."""
        df_list = []
        presence_mask_list = []
        for layer in self._selected_layers:
            # All layers in self._selected_layers are guaranteed to have features
            if layer.features is not None:
                if isinstance(layer.features, pd.DataFrame):
                    # Make a copy to avoid modifying the original layer.features
                    df = layer.features.copy()

                    # Only add 'Layer' column for multiple layers
                    if (
                        len(self._selected_layers) > 1
                        and 'Layer' not in df.columns
                    ):
                        df['Layer'] = layer.name
                        df['Layer'] = df['Layer'].astype('category')
                        # Move 'Layer' to the first column
                        cols = list(df.columns)
                        cols.remove('Layer')
                        cols.insert(0, 'Layer')
                        df = df[cols]
                    df_list.append(df)
                    presence_mask_list.append(~df.isna())
                else:
                    # TODO: Handle non-pandas dataframe libraries here
                    pass
        # Combine all dataframes and presence masks, preserving original indices per layer
        df = pd.concat(df_list, ignore_index=False, join=join)
        presence_mask = pd.concat(
            presence_mask_list, ignore_index=False, join=join
        )
        nan_cells_introduced_by_join = df.isna() & presence_mask.isna()
        # Convert columns with join-introduced NaNs to object dtype to ensure
        # pd.NA will be set instead of internally converted to NaT or np.nan
        cols_with_join_nans = nan_cells_introduced_by_join.any()
        for col in df.columns[cols_with_join_nans]:
            df[col] = df[col].astype('object')
        # Replace only NaNs introduced by outer join with pd.NA
        df = df.mask(nan_cells_introduced_by_join, pd.NA)
        return df

    def _on_editable_change(self):
        self.table.model().sourceModel().editable = self.toggle.isChecked()

    def _on_join_change(self):
        """Update the table when join mode changes."""
        self._on_features_change()

    def _get_original_categories_from_layers(
        self,
    ) -> dict[str, dict[str, list]]:
        """Extract categories by layer before concat loses dtype information.

        When concatenating DataFrames with columns that are categorical in some
        layers but strings (or missing) in others, pandas converts all to object dtype.
        This method captures the original categorical information per layer.

        Returns
        -------
        dict[str, dict[str, list]]
            Mapping: {layer_name: {column_name: [category_values]}}
            Only includes columns that were categorical in each layer.
        """
        categories_dict: dict[str, dict[str, list]] = {}
        for layer in self._selected_layers:
            if layer.features is not None and isinstance(
                layer.features, pd.DataFrame
            ):
                layer_categories: dict[str, list] = {}
                for col in layer.features.columns:
                    if isinstance(
                        layer.features[col].dtype, pd.CategoricalDtype
                    ):
                        layer_categories[col] = list(
                            layer.features[col].cat.categories
                        )
                if layer_categories:  # Only add if there are categoricals
                    categories_dict[layer.name] = layer_categories
        return categories_dict

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

        # Handle single layer case (no 'Layer' column)
        if len(self._selected_layers) == 1:
            layer = self._selected_layers[0]
            # All rows belong to this layer; global row indices are layer-specific indices
            layer_specific_indices = np.array(selected_global_rows)

            with self._block_selection():
                if hasattr(layer, 'selected_label'):
                    layer.selected_label = layer_specific_indices[-1]
                elif hasattr(layer, 'selected_data'):
                    layer.selected_data = set(layer_specific_indices)
            return

        # Handle multiple layers case (has source column)
        # Indices may not be continuous, so use positional indexing
        selections_by_layer = {}
        source_col_name = df.columns[
            0
        ]  # Source column is always first when present

        for row_pos in selected_global_rows:
            # Get source identifier for this row
            layer_name = df.iloc[row_pos][source_col_name]
            # Convert global row to layer-specific row
            layer_row_idx = self._global_row_to_layer_row(
                df, row_pos, layer_name
            )
            selections_by_layer.setdefault(layer_name, []).append(
                layer_row_idx
            )

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

        # Handle single layer case (no 'Layer' column)
        if len(self._selected_layers) == 1:
            layer = self._selected_layers[0]
            # All rows belong to this single layer
            # layer.selected_data/selected_label refer to positions in the layer

            if hasattr(layer, 'selected_label'):
                sel = layer.selected_label
                # Position in layer maps directly to df position
                indices += [[sel]]
            elif hasattr(layer, 'selected_data'):
                sel = layer.selected_data
                # Positions in layer map directly to df positions
                indices += [list(sel)]
        else:
            # Handle multiple sources case
            for layer in self._selected_layers:
                if hasattr(layer, 'selected_label'):
                    sel = layer.selected_label
                    # Position within the layer's features
                    global_row = self._layer_row_to_global_row(
                        df, sel, layer.name
                    )
                    if global_row is not None:
                        indices += [[global_row]]
                elif hasattr(layer, 'selected_data'):
                    sel = layer.selected_data
                    # Positions within the layer's features
                    for layer_pos in sel:
                        global_row = self._layer_row_to_global_row(
                            df, layer_pos, layer.name
                        )
                        if global_row is not None:
                            indices += [[global_row]]
                else:
                    continue

        if not indices:
            # deselect all rows if no selection attributes are found
            self.table.selectionModel().clearSelection()
            self.table.viewport().update()
            return  # nothing to select

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

    def _on_table_data_changed(self, topLeft, bottomRight, roles=None):
        """Sync edits in the table back to the correct layer.features DataFrame."""
        model = self.table.model().sourceModel()
        df = model.df
        # For each edited cell
        for row in range(topLeft.row(), bottomRight.row() + 1):
            # Determine which layer this row belongs to
            if len(self._selected_layers) == 1:
                # Single layer case: all rows belong to this layer
                layer = self._selected_layers[0]
                # Row position directly maps to layer position
                layer_row_idx = row
            else:
                # Multiple sources case: find source from the source column
                source_col_name = df.columns[
                    0
                ]  # Source column is always first when present
                layer_name = df.iloc[row][source_col_name]
                layer = next(
                    ly for ly in self._selected_layers if ly.name == layer_name
                )
                # Convert global row to layer-specific row
                layer_row_idx = self._global_row_to_layer_row(
                    df, row, layer_name
                )

            # Update the layer features DataFrame (except if immutable columns)
            for col in range(topLeft.column(), bottomRight.column() + 1):
                if model.is_column_immutable(col):
                    continue
                # Map view column to DataFrame column
                df_col_idx = model._get_df_col_idx(col)
                if df_col_idx is None:
                    continue
                col_name = df.columns[df_col_idx]
                if col_name not in layer.features.columns:
                    continue  # do not update features if column not present
                layer.features.iloc[
                    layer_row_idx, layer.features.columns.get_loc(col_name)
                ] = df.iloc[row, df_col_idx]

    @contextmanager
    def _block_selection(self):
        self._selection_blocked = True
        yield
        self._selection_blocked = False

    def _global_row_to_layer_row(
        self, df: pd.DataFrame, global_row: int, layer_name: str
    ) -> int:
        """Convert global DataFrame row position to layer-specific row position.

        Parameters
        ----------
        df : pd.DataFrame
            The combined features DataFrame
        global_row : int
            Row position in the combined DataFrame
        layer_name : str
            Name of the layer

        Returns
        -------
        int
            Row position within the layer's features
        """
        source_col_name = df.columns[0]
        matching_rows = df[df[source_col_name] == layer_name]
        layer_row_positions = matching_rows.index.tolist()
        df_row_index_value = df.index[global_row]

        # Find all occurrences of this index value in the layer
        layer_local_positions = [
            i
            for i, idx_val in enumerate(layer_row_positions)
            if idx_val == df_row_index_value
        ]

        # Handle duplicate indices by checking actual DataFrame position
        if len(layer_local_positions) == 1:
            return layer_local_positions[0]
        # Multiple rows with same index value in this layer
        layer_start_pos = df.index.tolist().index(layer_row_positions[0])
        return global_row - layer_start_pos

    def _layer_row_to_global_row(
        self, df: pd.DataFrame, layer_row: int, layer_name: str
    ) -> int | None:
        """Convert layer-specific row position to global DataFrame row position.

        Parameters
        ----------
        df : pd.DataFrame
            The combined features DataFrame
        layer_row : int
            Row position within the layer's features
        layer_name : str
            Name of the layer

        Returns
        -------
        int | None
            Row position in the combined DataFrame, or None if not found
        """
        source_col_name = df.columns[0]
        matching_rows = df[df[source_col_name] == layer_name]

        if layer_row >= len(matching_rows):
            return None

        # Get the index value at this layer position
        global_pos = matching_rows.iloc[[layer_row]].index.tolist()[0]

        # Find this index value's position in the full DataFrame
        df_positions = df.index.tolist()
        layer_positions = [
            i
            for i, idx in enumerate(df_positions)
            if idx == global_pos and df.iloc[i][source_col_name] == layer_name
        ]

        return layer_positions[0] if layer_positions else None

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
