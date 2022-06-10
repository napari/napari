from qtpy.QtWidgets import QDialog, QHBoxLayout, QLabel, QVBoxLayout

from ...utils.translations import trans


class QtLayerInfoDialog(QDialog):
    """Layer information dialog that can be accessed when right clicking
    layer on layer list."""

    # resized = Signal(QSize)

    def __init__(self, layer, parent=None):

        super().__init__(parent)

        self.setWindowTitle(trans._("Layer Information"))
        name = QLabel(layer.name)
        path = QLabel(layer.source.path)
        plugin = QLabel(layer.source.reader_plugin)
        sample = QLabel(str(layer.source.sample))
        widget = QLabel(str(layer.source.widget))

        # Layout
        left_layout = QVBoxLayout()
        left_layout.addWidget(QLabel('Name: '))
        left_layout.addWidget(QLabel('Path: '))
        left_layout.addWidget(QLabel('Reader plugin: '))
        left_layout.addWidget(QLabel('Sample: '))
        left_layout.addWidget(QLabel('Widget: '))

        right_layout = QVBoxLayout()
        right_layout.addWidget(name)
        right_layout.addWidget(path)
        right_layout.addWidget(plugin)
        right_layout.addWidget(sample)
        right_layout.addWidget(widget)

        final_layout = QHBoxLayout()
        final_layout.addLayout(left_layout)
        final_layout.addLayout(right_layout)

        self.setLayout(final_layout)
