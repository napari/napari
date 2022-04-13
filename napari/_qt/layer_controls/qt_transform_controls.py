from qtpy.QtWidgets import QComboBox, QFormLayout, QFrame, QLabel

from ...utils.translations import trans
from ..widgets.qt_spinbox import QtSpinBox


class QtTranslateDimControl(QtSpinBox):
    def __init__(self, layer, name):
        super().__init__()
        val_min = -90
        val_max = 90
        self.name = name

        self.setRange(val_min, val_max)


class QtTranslateTypeControl(QComboBox):
    def __init__(self, layer, name):
        super().__init__()
        self.name = name
        options = ['Option 1', 'Option 2', 'Option 3']
        self.addItems(options)


class QtScaleControl(QtSpinBox):
    def __init__(self, layer, name):
        super().__init__()

        # set up
        self.setRange(0, 50)
        self.name = name


class QtTransformControls(QFrame):
    def __init__(self, layer):
        super().__init__()

        translate_x_widget = QtTranslateDimControl(layer, 'Translate x')
        translate_y_widget = QtTranslateDimControl(layer, 'Translate y')
        translate_z_widget = QtTranslateDimControl(layer, 'Translate z')
        translate_type_widget = QtTranslateTypeControl(layer, 'Type')

        scale_widget = QtScaleControl(layer, 'Scale')

        form_layout = QFormLayout()

        form_layout.insertRow(
            0,
            QLabel(trans._(translate_x_widget.name + ':')),
            translate_x_widget,
        )
        form_layout.insertRow(
            1,
            QLabel(trans._(translate_y_widget.name + ':')),
            translate_y_widget,
        )
        form_layout.insertRow(
            2,
            QLabel(trans._(translate_z_widget.name + ':')),
            translate_z_widget,
        )
        form_layout.insertRow(
            3,
            QLabel(trans._(translate_type_widget.name + ':')),
            translate_type_widget,
        )
        form_layout.insertRow(
            4, QLabel(trans._(scale_widget.name + ':')), scale_widget
        )

        self.setLayout(form_layout)
