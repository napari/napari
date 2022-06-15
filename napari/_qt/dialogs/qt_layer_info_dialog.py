from qtpy.QtWidgets import QDialog, QFormLayout, QLabel

from ...utils.translations import trans


class QtLayerInfoDialog(QDialog):
    """Layer information dialog that can be accessed when right clicking
    layer on layer list."""

    # resized = Signal(QSize)

    def __init__(self, layer, parent=None):

        super().__init__(parent)

        self.setWindowTitle(trans._("Layer Information"))

        # Layout
        layout = QFormLayout()
        layout.addRow('Name: ', QLabel(layer.name))
        if layer.source.reader_plugin:
            layout.addRow(
                'Source plugin: ', QLabel(layer.source.reader_plugin)
            )
            layout.addRow('Path: ', QLabel(layer.source.path))
        if layer.source.sample:
            layout.addRow('Source sample: ', QLabel(layer.source.sample[0]))
        if layer.source.widget:
            layout.addRow(
                'Source widget: ',
                QLabel(str(layer.source.widget._function.__name__)),
            )

        self.setLayout(layout)
