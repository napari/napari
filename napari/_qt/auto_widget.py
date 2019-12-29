from enum import Enum

from qtpy.QtCore import Signal
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLineEdit,
    QSpinBox,
)


class TupleWidgetFrame(QFrame):
    """Special Container widget to represent tuples and lists"""

    valueChanged = Signal()

    def __init__(self, tup, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setFrameStyle(self.Panel | self.Raised)
        self._layout = QHBoxLayout()
        self.setLayout(self._layout)
        self._values = list(tup)
        self._layout.setContentsMargins(2, 2, 2, 2)
        self._setters = []
        for i, val in enumerate(tup):
            stuff = val_to_widget(val)
            if not stuff:
                continue
            widg, changed, getter, setter = stuff
            changed.connect(self.set_param(i, getter, type(val)))
            self._layout.addWidget(widg)
            self._setters.append(setter)

    def set_param(self, i, getter, dtype):
        """ update the parameter dict when the widg has changed """

        def func():
            self._values[i] = dtype(getter())
            self.valueChanged.emit()

        return func

    def value(self):
        return tuple(self._values)

    def setValue(self, value):
        if not (
            isinstance(value, (list, tuple))
            and len(value) == self._layout.count()
        ):
            raise ValueError("invalid arugment length to set TupleWidgetFrame")
        for i, v in enumerate(value):
            self._setters[i](v)


def val_to_widget(val, dtype=None):
    """generate a widget that works for a given value type.

    Returns a tuple:
        widg: the widget object
        signal: the signal to listen for when the object has changed
        getter: the getter function to retrieve the object value
        setter: the setter function to set the object value
    """
    dtype = dtype if dtype is not None else type(val)
    if issubclass(dtype, Enum) and isinstance(val, str):
        # make sure enum values are not strings
        val = dtype(val)
    if dtype == bool:
        widg = QCheckBox()
        widg.setChecked(val)
        setter = widg.setChecked
        changed = widg.stateChanged
        getter = widg.isChecked
    elif dtype == int:
        widg = QSpinBox()
        widg.setValue(val)
        setter = widg.setValue
        changed = widg.valueChanged
        getter = widg.value
    elif dtype == float:
        widg = QDoubleSpinBox()
        widg.setValue(val)
        setter = widg.setValue
        changed = widg.valueChanged
        getter = widg.value
    elif dtype == str:
        # 'file' is a special value that will create a browse button
        widg = QLineEdit(str(val))
        setter = widg.setText
        changed = widg.textChanged
        getter = widg.text
    elif isinstance(val, Enum):
        widg = QComboBox()
        [widg.addItem(option.value) for option in val.__class__]
        widg.setCurrentText(val.value)
        setter = widg.setCurrentText
        changed = widg.currentTextChanged
        getter = widg.currentText
    elif isinstance(val, (tuple, list)):
        widg = TupleWidgetFrame(val)
        setter = widg.setValue
        changed = widg.valueChanged
        getter = widg.value
    else:
        return None
    return widg, changed, getter, setter
