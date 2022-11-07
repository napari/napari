from functools import partial
from typing import Dict, List, Optional, TYPE_CHECKING, Tuple

from qtpy import QtCore, QtGui, QtWidgets

from napari._qt.widgets.qt_extension2reader import Extension2ReaderTable
from napari._qt.widgets.qt_highlight_preview import QtHighlightSizePreviewWidget
from napari._qt.widgets.qt_keyboard_settings import ShortcutEditor

from .signal import Signal
from .utils import is_concrete_schema, iter_layout_widgets, state_property

from napari._qt.widgets.qt_plugin_sorter import QtPluginSorter

from napari._qt.widgets.qt_spinbox import QtSpinBox


if TYPE_CHECKING:
    from .form import WidgetBuilder


class SchemaWidgetMixin:
    on_changed = Signal()

    VALID_COLOUR = '#ffffff'
    INVALID_COLOUR = '#f6989d'

    def __init__(
        self,
        schema: dict,
        ui_schema: dict,
        widget_builder: 'WidgetBuilder',
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.schema = schema
        self.ui_schema = ui_schema
        self.widget_builder = widget_builder

        self.on_changed.connect(lambda _: self.clear_error())
        self.configure()

    def configure(self):
        pass

    @state_property
    def state(self):
        raise NotImplementedError(f"{self.__class__.__name__}.state")

    @state.setter
    def state(self, state):
        raise NotImplementedError(f"{self.__class__.__name__}.state")

    def handle_error(self, path: Tuple[str], err: Exception):
        if path:
            raise ValueError("Cannot handle nested error by default")
        self._set_valid_state(err)

    def clear_error(self):
        self._set_valid_state(None)

    def _set_valid_state(self, error: Exception = None):
        palette = self.palette()
        colour = QtGui.QColor()
        colour.setNamedColor(
            self.VALID_COLOUR if error is None else self.INVALID_COLOUR
        )
        palette.setColor(self.backgroundRole(), colour)

        self.setPalette(palette)
        self.setToolTip("" if error is None else error.message)  # TODO


class TextSchemaWidget(SchemaWidgetMixin, QtWidgets.QLineEdit):
    def configure(self):
        self.textChanged.connect(self.on_changed.emit)
        self.opacity = QtWidgets.QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self.opacity)
        self.opacity.setOpacity(1)

    @state_property
    def state(self) -> str:
        return str(self.text())

    @state.setter
    def state(self, state: str):
        self.setText(state)

    def setDescription(self, description: str):
        self.description = description


class PasswordWidget(TextSchemaWidget):
    def configure(self):
        super().configure()

        self.setEchoMode(self.Password)

    def setDescription(self, description: str):
        self.description = description


class TextAreaSchemaWidget(SchemaWidgetMixin, QtWidgets.QTextEdit):
    @state_property
    def state(self) -> str:
        return str(self.toPlainText())

    @state.setter
    def state(self, state: str):
        self.setPlainText(state)

    def configure(self):
        self.textChanged.connect(lambda: self.on_changed.emit(self.state))
        self.opacity = QtWidgets.QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self.opacity)
        self.opacity.setOpacity(1)

    def setDescription(self, description: str):
        self.description = description


class CheckboxSchemaWidget(SchemaWidgetMixin, QtWidgets.QCheckBox):
    @state_property
    def state(self) -> bool:
        return self.isChecked()

    @state.setter
    def state(self, checked: bool):
        self.setChecked(checked)

    def configure(self):
        self.stateChanged.connect(lambda _: self.on_changed.emit(self.state))
        self.opacity = QtWidgets.QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self.opacity)
        self.opacity.setOpacity(1)

    def setDescription(self, description: str):
        self.description = description


class SpinDoubleSchemaWidget(SchemaWidgetMixin, QtWidgets.QDoubleSpinBox):
    @state_property
    def state(self) -> float:
        return self.value()

    @state.setter
    def state(self, state: float):
        self.setValue(state)

    def configure(self):
        self.valueChanged.connect(self.on_changed.emit)
        self.opacity = QtWidgets.QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self.opacity)
        self.opacity.setOpacity(1)

    def setDescription(self, description: str):
        self.description = description


class PluginWidget(SchemaWidgetMixin, QtPluginSorter):
    @state_property
    def state(self) -> int:
        return self.value()

    @state.setter
    def state(self, state: int):
        return None
        # self.setValue(state)

    def configure(self):
        self.hook_list.order_changed.connect(self.on_changed.emit)

    def setDescription(self, description: str):
        self.description = description


class SpinSchemaWidget(SchemaWidgetMixin, QtSpinBox):
    @state_property
    def state(self) -> int:
        return self.value()

    @state.setter
    def state(self, state: int):
        self.setValue(state)

    def configure(self):
        self.valueChanged.connect(self.on_changed.emit)
        self.opacity = QtWidgets.QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self.opacity)
        self.opacity.setOpacity(1)

        minimum = -2147483648
        if "minimum" in self.schema:
            minimum = self.schema["minimum"]
            if self.schema.get("exclusiveMinimum"):
                minimum += 1

        maximum = 2147483647
        if "maximum" in self.schema:
            maximum = self.schema["maximum"]
            if self.schema.get("exclusiveMaximum"):
                maximum -= 1

        self.setRange(minimum, maximum)

        if "not" in self.schema and 'const' in self.schema["not"]:
            self.setProhibitValue(self.schema["not"]['const'])

    def setDescription(self, description: str):
        self.description = description


class IntegerRangeSchemaWidget(SchemaWidgetMixin, QtWidgets.QSlider):
    def __init__(
        self,
        schema: dict,
        ui_schema: dict,
        widget_builder: 'WidgetBuilder',
    ):
        super().__init__(
            schema, ui_schema, widget_builder, orientation=QtCore.Qt.Horizontal
        )

    @state_property
    def state(self) -> int:
        return self.value()

    @state.setter
    def state(self, state: int):
        self.setValue(state)

    def configure(self):
        self.valueChanged.connect(self.on_changed.emit)
        self.opacity = QtWidgets.QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self.opacity)
        self.opacity.setOpacity(1)

        minimum = 0
        if "minimum" in self.schema:
            minimum = self.schema["minimum"]
            if self.schema.get("exclusiveMinimum"):
                minimum += 1

        maximum = 0
        if "maximum" in self.schema:
            maximum = self.schema["maximum"]
            if self.schema.get("exclusiveMaximum"):
                maximum -= 1

        if "multipleOf" in self.schema:
            self.setTickInterval(self.schema["multipleOf"])
            self.setSingleStep(self.schema["multipleOf"])
            self.setTickPosition(self.TicksBothSides)

        self.setRange(minimum, maximum)

    def setDescription(self, description: str):
        self.description = description


class QColorButton(QtWidgets.QPushButton):
    """Color picker widget QPushButton subclass.

    Implementation derived from https://martinfitzpatrick.name/article/qcolorbutton-a-color-selector-tool-for-pyqt/
    """

    colorChanged = QtCore.Signal()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._color = None
        self.pressed.connect(self.onColorPicker)
        self.opacity = QtWidgets.QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self.opacity)
        self.opacity.setOpacity(1)

    def color(self):
        return self._color

    def setColor(self, color):
        if color != self._color:
            self._color = color
            self.colorChanged.emit()

        if self._color:
            self.setStyleSheet("background-color: %s;" % self._color)
        else:
            self.setStyleSheet("")

    def onColorPicker(self):
        dlg = QtWidgets.QColorDialog(self)
        if self._color:
            dlg.setCurrentColor(QtGui.QColor(self._color))

        if dlg.exec_():
            self.setColor(dlg.currentColor().name())

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.RightButton:
            self.setColor(None)

        return super().mousePressEvent(event)

    def setDescription(self, description: str):
        self.description = description


class ColorSchemaWidget(SchemaWidgetMixin, QColorButton):
    """Widget representation of a string with the 'color' format keyword."""

    def configure(self):
        self.colorChanged.connect(lambda: self.on_changed.emit(self.state))
        self.opacity = QtWidgets.QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self.opacity)
        self.opacity.setOpacity(1)

    @state_property
    def state(self) -> str:
        return self.color()

    @state.setter
    def state(self, data: str):
        self.setColor(data)

    def setDescription(self, description: str):
        self.description = description


class FilepathSchemaWidget(SchemaWidgetMixin, QtWidgets.QWidget):
    def __init__(
        self,
        schema: dict,
        ui_schema: dict,
        widget_builder: 'WidgetBuilder',
    ):
        super().__init__(schema, ui_schema, widget_builder)

        self.opacity = QtWidgets.QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self.opacity)
        self.opacity.setOpacity(1)

        layout = QtWidgets.QHBoxLayout()
        self.setLayout(layout)

        self.path_widget = QtWidgets.QLineEdit()
        self.button_widget = QtWidgets.QPushButton("Browse")
        layout.addWidget(self.path_widget)
        layout.addWidget(self.button_widget)

        self.button_widget.clicked.connect(self._on_clicked)
        self.path_widget.textChanged.connect(self.on_changed.emit)

    def _on_clicked(self, flag):
        path, filter = QtWidgets.QFileDialog.getOpenFileName()
        self.path_widget.setText(path)

    @state_property
    def state(self) -> str:
        return self.path_widget.text()

    @state.setter
    def state(self, state: str):
        self.path_widget.setText(state)

    def setDescription(self, description: str):
        self.description = description


class ArrayControlsWidget(QtWidgets.QWidget):
    on_delete = QtCore.Signal()
    on_move_up = QtCore.Signal()
    on_move_down = QtCore.Signal()

    def __init__(self):
        super().__init__()

        style = self.style()

        self.up_button = QtWidgets.QPushButton()
        self.up_button.setIcon(style.standardIcon(QtWidgets.QStyle.SP_ArrowUp))
        self.up_button.clicked.connect(lambda _: self.on_move_up.emit())

        self.opacity = QtWidgets.QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self.opacity)
        self.opacity.setOpacity(1)

        self.delete_button = QtWidgets.QPushButton()
        self.delete_button.setIcon(
            style.standardIcon(QtWidgets.QStyle.SP_DialogCancelButton)
        )
        self.delete_button.clicked.connect(lambda _: self.on_delete.emit())

        self.down_button = QtWidgets.QPushButton()
        self.down_button.setIcon(
            style.standardIcon(QtWidgets.QStyle.SP_ArrowDown)
        )
        self.down_button.clicked.connect(lambda _: self.on_move_down.emit())

        group_layout = QtWidgets.QHBoxLayout()
        self.setLayout(group_layout)
        group_layout.addWidget(self.up_button)
        group_layout.addWidget(self.down_button)
        group_layout.addWidget(self.delete_button)
        group_layout.setSpacing(0)
        group_layout.addStretch(0)

    def setDescription(self, description: str):
        self.description = description


class ArrayRowWidget(QtWidgets.QWidget):
    def __init__(
        self, widget: QtWidgets.QWidget, controls: ArrayControlsWidget
    ):
        super().__init__()

        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(widget)
        layout.addWidget(controls)
        self.setLayout(layout)

        self.opacity = QtWidgets.QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self.opacity)
        self.opacity.setOpacity(1)

        self.widget = widget
        self.controls = controls

    def setDescription(self, description: str):
        self.description = description


class ArraySchemaWidget(SchemaWidgetMixin, QtWidgets.QWidget):
    @property
    def rows(self) -> List[ArrayRowWidget]:
        return [*iter_layout_widgets(self.array_layout)]

    @state_property
    def state(self) -> list:
        return [r.widget.state for r in self.rows]

    @state.setter
    def state(self, state: list):
        for row in self.rows:
            self._remove_item(row)

        for item in state:
            self._add_item(item)

        self.on_changed.emit(self.state)

    def handle_error(self, path: Tuple[str], err: Exception):
        index, *tail = path
        self.rows[index].widget.handle_error(tail, err)

    def configure(self):
        layout = QtWidgets.QVBoxLayout()
        style = self.style()

        self.opacity = QtWidgets.QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self.opacity)
        self.opacity.setOpacity(1)

        self.add_button = QtWidgets.QPushButton()
        self.add_button.setIcon(
            style.standardIcon(QtWidgets.QStyle.SP_FileIcon)
        )
        self.add_button.clicked.connect(lambda _: self.add_item())

        self.array_layout = QtWidgets.QVBoxLayout()
        array_widget = QtWidgets.QWidget(self)
        array_widget.setLayout(self.array_layout)

        self.on_changed.connect(self._on_updated)

        layout.addWidget(self.add_button)
        layout.addWidget(array_widget)
        self.setLayout(layout)

    def _on_updated(self, state):
        # Update add button
        disabled = self.next_item_schema is None
        self.add_button.setEnabled(not disabled)

        previous_row = None
        for i, row in enumerate(self.rows):
            if previous_row:
                can_exchange_previous = (
                    previous_row.widget.schema == row.widget.schema
                )
                row.controls.up_button.setEnabled(can_exchange_previous)
                previous_row.controls.down_button.setEnabled(
                    can_exchange_previous
                )
            else:
                row.controls.up_button.setEnabled(False)
            row.controls.delete_button.setEnabled(not self.is_fixed_schema(i))
            previous_row = row

        if previous_row:
            previous_row.controls.down_button.setEnabled(False)

    def is_fixed_schema(self, index: int) -> bool:
        schema = self.schema['items']
        if isinstance(schema, dict):
            return False

        return index < len(schema)

    @property
    def next_item_schema(self) -> Optional[dict]:
        item_schema = self.schema['items']

        if isinstance(item_schema, dict):
            return item_schema

        index = len(self.rows)

        try:
            item_schema = item_schema[index]
        except IndexError:
            item_schema = self.schema.get("additionalItems", {})
            if isinstance(item_schema, bool):
                return None

        if not is_concrete_schema(item_schema):
            return None

        return item_schema

    def add_item(self, item_state=None):
        self._add_item(item_state)
        self.on_changed.emit(self.state)

    def remove_item(self, row: ArrayRowWidget):
        self._remove_item(row)
        self.on_changed.emit(self.state)

    def move_item_up(self, row: ArrayRowWidget):
        index = self.rows.index(row)
        self.array_layout.insertWidget(max(0, index - 1), row)
        self.on_changed.emit(self.state)

    def move_item_down(self, row: ArrayRowWidget):
        index = self.rows.index(row)
        self.array_layout.insertWidget(min(len(self.rows) - 1, index + 1), row)
        self.on_changed.emit(self.state)

    def _add_item(self, item_state=None):
        item_schema = self.next_item_schema

        # Create widget
        item_ui_schema = self.ui_schema.get("items", {})
        widget = self.widget_builder.create_widget(
            item_schema, item_ui_schema, item_state
        )
        controls = ArrayControlsWidget()

        # Create row
        row = ArrayRowWidget(widget, controls)
        self.array_layout.addWidget(row)

        # Setup callbacks
        widget.on_changed.connect(partial(self.widget_on_changed, row))
        controls.on_delete.connect(partial(self.remove_item, row))
        controls.on_move_up.connect(partial(self.move_item_up, row))
        controls.on_move_down.connect(partial(self.move_item_down, row))

        return row

    def _remove_item(self, row: ArrayRowWidget):
        self.array_layout.removeWidget(row)
        row.deleteLater()

    def widget_on_changed(self, row: ArrayRowWidget, value):
        self.state[self.rows.index(row)] = value
        self.on_changed.emit(self.state)


class HighlightSizePreviewWidget(
    SchemaWidgetMixin, QtHighlightSizePreviewWidget
):
    @state_property
    def state(self) -> int:
        return self.value()

    def setDescription(self, description: str):
        self._description.setText(description)

    @state.setter
    def state(self, state: int):
        self.setValue(state)

    def configure(self):
        self.valueChanged.connect(self.on_changed.emit)
        self.opacity = QtWidgets.QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self.opacity)
        self.opacity.setOpacity(1)


class ShortcutsWidget(SchemaWidgetMixin, ShortcutEditor):
    @state_property
    def state(self) -> dict:
        return self.value()

    def setDescription(self, description: str):
        self.description = description

    @state.setter
    def state(self, state: dict):
        # self.setValue(state)
        return None

    def configure(self):
        self.valueChanged.connect(self.on_changed.emit)
        self.opacity = QtWidgets.QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self.opacity)
        self.opacity.setOpacity(1)


class Extension2ReaderWidget(SchemaWidgetMixin, Extension2ReaderTable):
    @state_property
    def state(self) -> dict:
        return self.value()

    def setDescription(self, description: str):
        self.description = description

    @state.setter
    def state(self, state: dict):
        # self.setValue(state)
        return None

    def configure(self):
        self.valueChanged.connect(self.on_changed.emit)
        self.opacity = QtWidgets.QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self.opacity)
        self.opacity.setOpacity(1)

class ObjectSchemaWidget(SchemaWidgetMixin, QtWidgets.QGroupBox):
    def __init__(
        self,
        schema: dict,
        ui_schema: dict,
        widget_builder: 'WidgetBuilder',
    ):
        super().__init__(schema, ui_schema, widget_builder)

        self.widgets = self.populate_from_schema(
            schema, ui_schema, widget_builder
        )

    @state_property
    def state(self) -> dict:
        return {k: w.state for k, w in self.widgets.items()}

    @state.setter
    def state(self, state: dict):
        for name, value in state.items():
            self.widgets[name].state = value

    def handle_error(self, path: Tuple[str], err: Exception):
        name, *tail = path
        self.widgets[name].handle_error(tail, err)

    def widget_on_changed(self, name: str, value):
        self.state[name] = value
        self.on_changed.emit(self.state)

    def setDescription(self, description: str):
        self.description = description

    def populate_from_schema(
        self,
        schema: dict,
        ui_schema: dict,
        widget_builder: 'WidgetBuilder',
    ) -> Dict[str, QtWidgets.QWidget]:
        layout = QtWidgets.QFormLayout()
        self.setLayout(layout)
        layout.setAlignment(QtCore.Qt.AlignTop)
        self.setFlat(False)

        if 'title' in schema:
            self.setTitle(schema['title'])


        # Populate rows
        widgets = {}
        layout.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy(1))
        for name, sub_schema in schema['properties'].items():
            if 'description' in sub_schema:
                description = sub_schema['description']
            else:
                description = ""

            sub_ui_schema = ui_schema.get(name, {})
            widget = widget_builder.create_widget(
                sub_schema, sub_ui_schema, description=description
            )  # TODO onchanged
            widget._name = name
            widget.on_changed.connect(partial(self.widget_on_changed, name))
            label = sub_schema.get("title", name)
            layout.addRow(label, widget)
            widgets[name] = widget

        return widgets


class EnumSchemaWidget(SchemaWidgetMixin, QtWidgets.QComboBox):
    @state_property
    def state(self):
        return self.itemData(self.currentIndex())

    @state.setter
    def state(self, value):
        index = self.findData(value)
        if index == -1:
            raise ValueError(value)
        self.setCurrentIndex(index)

    def configure(self):
        options = self.schema["enum"]
        for i, opt in enumerate(options):
            self.addItem(str(opt))
            self.setItemData(i, opt)

        self.currentIndexChanged.connect(
            lambda _: self.on_changed.emit(self.state)
        )

        self.opacity = QtWidgets.QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self.opacity)
        self.opacity.setOpacity(1)

    def _index_changed(self, index: int):
        self.on_changed.emit(self.state)

    def setDescription(self, description: str):
        self.description = description


class FormWidget(QtWidgets.QWidget):
    def __init__(self, widget: SchemaWidgetMixin):
        super().__init__()
        layout = QtWidgets.QVBoxLayout()
        self.setLayout(layout)

        self.error_widget = QtWidgets.QGroupBox()
        self.error_widget.setTitle("Errors")
        self.error_layout = QtWidgets.QVBoxLayout()
        self.error_widget.setLayout(self.error_layout)
        self.error_widget.hide()

        layout.addWidget(self.error_widget)
        layout.addWidget(widget)

        self.widget = widget

    def display_errors(self, errors: List[Exception]):
        self.error_widget.show()

        layout = self.error_widget.layout()
        while True:
            item = layout.takeAt(0)
            if not item:
                break
            item.widget().deleteLater()

        for err in errors:
            widget = QtWidgets.QLabel(
                f"<b>.{'.'.join(err.path)}</b> {err.message}"
            )
            layout.addWidget(widget)

    def clear_errors(self):
        self.error_widget.hide()
